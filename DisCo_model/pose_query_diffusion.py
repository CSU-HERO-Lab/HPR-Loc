import math
from typing import Dict, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from DisCo_model.image_patch_encoder import ImagePatchEncoder
from DisCo_model.floorplan_encoder import ResNetFloorplanEncoder


LEGACY_COORDINATE_CONVENTION = "legacy_normalized_v0"
METRIC_COORDINATE_CONVENTION = "metric_cell_center_v1"
SUPPORTED_COORDINATE_CONVENTIONS = {
    LEGACY_COORDINATE_CONVENTION,
    METRIC_COORDINATE_CONVENTION,
}


def map_xy_to_normalized(xy: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
    """Map continuous pixel coordinates to the normalized map extent.

    Pose coordinates are measured from the top-left map boundary, so 0 and W/H
    map to -1 and +1. Feature tokens use cell centers within the same extent.
    """
    wh = wh.to(device=xy.device, dtype=xy.dtype)
    while wh.ndim < xy.ndim:
        wh = wh.unsqueeze(1)
    return xy / wh.clamp_min(1.0) * 2.0 - 1.0


def normalized_to_map_xy(xy_norm: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`map_xy_to_normalized`."""
    wh = wh.to(device=xy_norm.device, dtype=xy_norm.dtype)
    while wh.ndim < xy_norm.ndim:
        wh = wh.unsqueeze(1)
    return (xy_norm + 1.0) * 0.5 * wh


def build_cell_center_coordinates(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalized feature-cell centers in row-major order."""
    pos_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    pos_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    pos_x = pos_x * 2.0 - 1.0
    pos_y = pos_y * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((steps / num_steps + s) / (1 + s)) * math.pi * 0.5
    ).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.feature_dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            / max(half_dim - 1, 1)
        )
        angles = timesteps.float().unsqueeze(-1) * frequencies
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.feature_dim:
            embedding = F.pad(embedding, (0, self.feature_dim - embedding.shape[-1]))
        return self.mlp(embedding)


class ImageConditionedMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_dim = int(config.get("diffusion_feature_dim", 128))
        self.coordinate_convention = config.get(
            "diffusion_coordinate_convention",
            LEGACY_COORDINATE_CONVENTION,
        )
        self.image_token_grid = tuple(config.get("image_token_grid", [6, 40]))
        self.freeze_image_backbone = bool(config.get("freeze_image_backbone", True))
        num_heads = int(config.get("diffusion_num_heads", 4))

        self.image_encoder = ImagePatchEncoder(
            encoder=config.get("image_encoder", "vits"),
            feature_dim=self.feature_dim,
            target_size=self.image_token_grid,
            checkpoint_path=config.get(
                "dptv2_ckpt_path", "checkpoints/depth_anything_v2_vits.pth"
            ),
            freeze_backbone=self.freeze_image_backbone,
            use_cls_token=False,
        )
        self.image_norm = nn.LayerNorm(self.feature_dim)
        image_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=num_heads,
            dim_feedforward=self.feature_dim * 4,
            dropout=float(config.get("diffusion_dropout", 0.1)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.image_mixer = nn.TransformerEncoder(
            image_layer,
            num_layers=int(config.get("diffusion_image_layers", 1)),
            norm=nn.LayerNorm(self.feature_dim),
        )

        map_feature_dim = int(config.get("diffusion_map_feature_dim", 64))
        self.map_encoder = ResNetFloorplanEncoder(
            feature_dim=map_feature_dim,
            input_mode=config["diffusion_map_input_mode"],
            context_blocks=int(config.get("diffusion_map_context_blocks", 2)),
            pretrained=bool(config.get("diffusion_map_pretrained", False)),
        )
        self.map_projection = nn.Conv2d(
            map_feature_dim, self.feature_dim, kernel_size=1, bias=False
        )
        self.map_pos_mlp = nn.Sequential(
            nn.Linear(2, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.map_norm = nn.LayerNorm(self.feature_dim)
        self.map_image_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            dropout=float(config.get("diffusion_dropout", 0.1)),
            batch_first=True,
        )
        self.map_image_norm = nn.LayerNorm(self.feature_dim)
        self.map_ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 4),
            nn.GELU(),
            nn.Dropout(float(config.get("diffusion_dropout", 0.1))),
            nn.Linear(self.feature_dim * 4, self.feature_dim),
        )
        self.map_ffn_norm = nn.LayerNorm(self.feature_dim)

    @staticmethod
    def build_map_coordinates(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        coordinate_convention: str = METRIC_COORDINATE_CONVENTION,
    ) -> torch.Tensor:
        if coordinate_convention == METRIC_COORDINATE_CONVENTION:
            return build_cell_center_coordinates(height, width, device, dtype)
        if coordinate_convention == LEGACY_COORDINATE_CONVENTION:
            pos_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
            pos_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
            return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        raise ValueError(
            f"Unsupported diffusion coordinate convention: {coordinate_convention!r}."
        )

    def forward(
        self,
        obs_img: torch.Tensor,
        floorplan_img: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_tokens = self.image_encoder(obs_img)
        image_tokens = self.image_mixer(self.image_norm(image_tokens))

        map_features = self.map_projection(self.map_encoder(floorplan_img))
        batch_size, channels, height, width = map_features.shape
        map_tokens = map_features.flatten(2).transpose(1, 2)
        map_coordinates = self.build_map_coordinates(
            height,
            width,
            map_tokens.device,
            map_tokens.dtype,
            self.coordinate_convention,
        )
        map_tokens = map_tokens + self.map_pos_mlp(map_coordinates).unsqueeze(0)
        map_tokens = self.map_norm(map_tokens)

        attended_map, _ = self.map_image_attn(
            query=map_tokens,
            key=image_tokens,
            value=image_tokens,
            need_weights=False,
        )
        map_tokens = self.map_image_norm(map_tokens + attended_map)
        map_tokens = self.map_ffn_norm(map_tokens + self.map_ffn(map_tokens))
        image_global = image_tokens.mean(dim=1)
        return map_tokens, map_coordinates, image_global


class PoseMapCrossAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        dropout: float,
        metric_relative_max_m: float,
        metric_relative_scale_m: float,
        coordinate_convention: str,
    ):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.metric_relative_max_m = float(metric_relative_max_m)
        self.metric_relative_scale_m = float(metric_relative_scale_m)
        self.coordinate_convention = coordinate_convention
        if self.metric_relative_max_m <= 0:
            raise ValueError("metric_relative_max_m must be positive.")
        if self.metric_relative_scale_m <= 0:
            raise ValueError("metric_relative_scale_m must be positive.")
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.output = nn.Linear(feature_dim, feature_dim)
        self.relative_bias = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, num_heads),
        )
        self.dropout = nn.Dropout(dropout)

    def project_map_tokens(
        self,
        map_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_map_tokens, feature_dim = map_tokens.shape
        key = self.key(map_tokens).view(
            batch_size, num_map_tokens, self.num_heads, self.head_dim
        )
        value = self.value(map_tokens).view(
            batch_size, num_map_tokens, self.num_heads, self.head_dim
        )
        if feature_dim != self.num_heads * self.head_dim:
            raise ValueError("Unexpected map-token feature dimension.")
        return key, value

    @staticmethod
    def build_metric_relative_features(
        noisy_pose: torch.Tensor,
        map_coordinates: torch.Tensor,
        wh: torch.Tensor,
        map_res,
        max_distance_m: float,
        scale_m: float,
    ) -> torch.Tensor:
        """Build clipped local-frame metric offsets with a fixed metric scale."""
        batch_size, num_particles = noisy_pose.shape[:2]
        num_map_tokens = map_coordinates.shape[0]
        wh = wh.to(device=noisy_pose.device, dtype=noisy_pose.dtype)
        map_res = torch.as_tensor(
            map_res,
            device=noisy_pose.device,
            dtype=noisy_pose.dtype,
        )
        if map_res.ndim == 0:
            map_res = map_res.expand(batch_size)
        map_res = map_res.reshape(batch_size, 1, 1)

        pose_xy = noisy_pose[..., :2]
        angle_vec = F.normalize(noisy_pose[..., 2:4], dim=-1, eps=1e-6)
        sin_theta = angle_vec[..., 0].unsqueeze(-1)
        cos_theta = angle_vec[..., 1].unsqueeze(-1)
        delta_norm = (
            map_coordinates.view(1, 1, num_map_tokens, 2)
            - pose_xy.unsqueeze(2)
        )
        delta_m = delta_norm * wh.view(batch_size, 1, 1, 2) * 0.5
        delta_m = delta_m * map_res.unsqueeze(-1)
        dx_m = delta_m[..., 0]
        dy_m = delta_m[..., 1]
        local_x_m = cos_theta * dx_m + sin_theta * dy_m
        local_y_m = -sin_theta * dx_m + cos_theta * dy_m
        distance_m = torch.sqrt(dx_m.square() + dy_m.square() + 1e-8)

        clip = float(max_distance_m)
        scale = float(scale_m)
        local_x = local_x_m.clamp(-clip, clip) / scale
        local_y = local_y_m.clamp(-clip, clip) / scale
        distance = distance_m.clamp(0.0, clip) / scale
        return torch.stack(
            [local_x, local_y, distance, distance.square()],
            dim=-1,
        )

    def forward(
        self,
        pose_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        noisy_pose: torch.Tensor,
        map_coordinates: torch.Tensor,
        wh: torch.Tensor,
        map_res,
        map_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, num_particles, feature_dim = pose_tokens.shape
        num_map_tokens = map_tokens.shape[1]

        query = self.query(pose_tokens).view(
            batch_size, num_particles, self.num_heads, self.head_dim
        )
        if map_key_value is None:
            key, value = self.project_map_tokens(map_tokens)
        else:
            key, value = map_key_value
            expected_shape = (
                batch_size,
                num_map_tokens,
                self.num_heads,
                self.head_dim,
            )
            if key.shape != expected_shape or value.shape != expected_shape:
                raise ValueError(
                    "Cached map K/V shape does not match the current map tokens."
                )
        attention = torch.einsum("bmhd,bnhd->bhmn", query, key) * self.scale

        if self.coordinate_convention == METRIC_COORDINATE_CONVENTION:
            relative_features = self.build_metric_relative_features(
                noisy_pose,
                map_coordinates,
                wh,
                map_res,
                self.metric_relative_max_m,
                self.metric_relative_scale_m,
            )
        else:
            pose_xy = noisy_pose[..., :2]
            angle_vec = F.normalize(noisy_pose[..., 2:4], dim=-1, eps=1e-6)
            sin_theta = angle_vec[..., 0].unsqueeze(-1)
            cos_theta = angle_vec[..., 1].unsqueeze(-1)
            delta = (
                map_coordinates.view(1, 1, num_map_tokens, 2)
                - pose_xy.unsqueeze(2)
            )
            dx = delta[..., 0]
            dy = delta[..., 1]
            local_x = cos_theta * dx + sin_theta * dy
            local_y = -sin_theta * dx + cos_theta * dy
            distance = torch.sqrt(dx.square() + dy.square() + 1e-8)
            relative_features = torch.stack(
                [local_x, local_y, distance, distance.square()], dim=-1
            )
        relative_bias = self.relative_bias(relative_features).permute(0, 3, 1, 2)
        attention = torch.softmax(attention + relative_bias, dim=-1)
        attention = self.dropout(attention)

        attended = torch.einsum("bhmn,bnhd->bmhd", attention, value)
        attended = attended.reshape(batch_size, num_particles, feature_dim)
        return self.output(attended)


class PoseDenoiserBlock(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        dropout: float,
        metric_relative_max_m: float,
        metric_relative_scale_m: float,
        coordinate_convention: str,
    ):
        super().__init__()
        self.cross_attn = PoseMapCrossAttention(
            feature_dim,
            num_heads,
            dropout,
            metric_relative_max_m,
            metric_relative_scale_m,
            coordinate_convention,
        )
        self.cross_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pose_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
        noisy_pose: torch.Tensor,
        map_coordinates: torch.Tensor,
        wh: torch.Tensor,
        map_res,
        map_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        attended = self.cross_attn(
            pose_tokens,
            map_tokens,
            noisy_pose,
            map_coordinates,
            wh,
            map_res,
            map_key_value=map_key_value,
        )
        pose_tokens = self.cross_norm(pose_tokens + self.dropout(attended))
        return self.ffn_norm(pose_tokens + self.dropout(self.ffn(pose_tokens)))


class PoseQueryDenoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_dim = int(config.get("diffusion_feature_dim", 128))
        self.num_steps = int(config.get("diffusion_train_steps", 1000))
        self.fourier_bands = int(config.get("diffusion_pose_fourier_bands", 4))
        pose_input_dim = 4 * (1 + 2 * self.fourier_bands)
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_input_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.time_embedding = SinusoidalTimeEmbedding(self.feature_dim)
        self.image_global_projection = nn.Linear(self.feature_dim, self.feature_dim)
        metric_relative_max_m = float(
            config.get("diffusion_metric_relative_max_m", 20.0)
        )
        metric_relative_scale_m = float(
            config.get("diffusion_metric_relative_scale_m", 5.0)
        )
        coordinate_convention = config.get(
            "diffusion_coordinate_convention",
            LEGACY_COORDINATE_CONVENTION,
        )
        self.blocks = nn.ModuleList(
            [
                PoseDenoiserBlock(
                    feature_dim=self.feature_dim,
                    num_heads=int(config.get("diffusion_num_heads", 4)),
                    dropout=float(config.get("diffusion_dropout", 0.1)),
                    metric_relative_max_m=metric_relative_max_m,
                    metric_relative_scale_m=metric_relative_scale_m,
                    coordinate_convention=coordinate_convention,
                )
                for _ in range(int(config.get("diffusion_denoiser_layers", 2)))
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 4),
        )

    def fourier_encode(self, pose: torch.Tensor) -> torch.Tensor:
        features = [pose]
        for band in range(self.fourier_bands):
            frequency = math.pi * (2**band)
            features.extend(
                [torch.sin(frequency * pose), torch.cos(frequency * pose)]
            )
        return torch.cat(features, dim=-1)

    def build_map_kv_cache(
        self,
        map_tokens: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            block.cross_attn.project_map_tokens(map_tokens)
            for block in self.blocks
        )

    def forward(
        self,
        noisy_pose: torch.Tensor,
        timesteps: torch.Tensor,
        map_tokens: torch.Tensor,
        map_coordinates: torch.Tensor,
        image_global: torch.Tensor,
        wh: torch.Tensor,
        map_res,
        map_kv_cache: Optional[
            Sequence[Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> torch.Tensor:
        if map_kv_cache is not None and len(map_kv_cache) != len(self.blocks):
            raise ValueError("Map K/V cache must contain one entry per block.")
        pose_tokens = self.pose_mlp(self.fourier_encode(noisy_pose))
        pose_tokens = pose_tokens + self.time_embedding(timesteps)
        pose_tokens = pose_tokens + self.image_global_projection(image_global).unsqueeze(1)
        for block_index, block in enumerate(self.blocks):
            pose_tokens = block(
                pose_tokens,
                map_tokens,
                noisy_pose,
                map_coordinates,
                wh,
                map_res,
                map_key_value=(
                    map_kv_cache[block_index]
                    if map_kv_cache is not None
                    else None
                ),
            )
        return self.output(pose_tokens)


class PoseQueryDiffusionLocalizer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.coordinate_convention = config.get(
            "diffusion_coordinate_convention",
            LEGACY_COORDINATE_CONVENTION,
        )
        if self.coordinate_convention not in SUPPORTED_COORDINATE_CONVENTIONS:
            raise ValueError(
                "Unsupported diffusion coordinate convention: "
                f"{self.coordinate_convention!r}."
            )
        self.map_res = float(config["datasets"].get("map_res", 0.02))
        self.num_train_steps = int(config.get("diffusion_train_steps", 1000))
        self.train_particles = int(config.get("diffusion_train_particles", 8))
        self.val_particles = int(config.get("diffusion_val_particles", 64))
        self.sample_steps = int(config.get("diffusion_sample_steps", 10))
        self.theta_loss_weight = float(config.get("diffusion_theta_loss_weight", 1.0))
        self.mode_sigma_m = float(config.get("diffusion_mode_sigma_m", 0.75))
        self.mode_sigma_deg = float(config.get("diffusion_mode_sigma_deg", 20.0))

        self.condition_encoder = ImageConditionedMapEncoder(config)
        self.denoiser = PoseQueryDenoiser(config)

        betas = cosine_beta_schedule(self.num_train_steps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            (1.0 - alpha_cumprod).sqrt(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.condition_encoder.freeze_image_backbone:
            self.condition_encoder.image_encoder.backbone.eval()
        return self

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return values[timesteps].unsqueeze(-1)

    @staticmethod
    def constrain_pose_state(pose_state: torch.Tensor) -> torch.Tensor:
        xy = pose_state[..., :2].clamp(-1.0, 1.0)
        angle = F.normalize(pose_state[..., 2:4], dim=-1, eps=1e-6)
        return torch.cat([xy, angle], dim=-1)

    def encode_pose(self, pose: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
        xy = map_xy_to_normalized(pose[:, :2], wh)
        angle = torch.stack([torch.sin(pose[:, 2]), torch.cos(pose[:, 2])], dim=-1)
        return torch.cat([xy, angle], dim=-1)

    def decode_pose(self, pose_state: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
        pose_state = self.constrain_pose_state(pose_state)
        xy = normalized_to_map_xy(pose_state[..., :2], wh)
        theta = torch.remainder(
            torch.atan2(pose_state[..., 2], pose_state[..., 3]),
            2 * math.pi,
        )
        return torch.cat([xy, theta.unsqueeze(-1)], dim=-1)

    def q_sample(
        self,
        x0_pose: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alpha_cumprod, timesteps) * x0_pose
            + self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps) * noise
        )

    def predict_x0_pose(
        self,
        noisy_pose: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        x0_pose = (
            noisy_pose
            - self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps)
            * predicted_noise
        ) / self._extract(self.sqrt_alpha_cumprod, timesteps).clamp_min(1e-6)
        return self.constrain_pose_state(x0_pose)

    def diffusion_loss(
        self,
        pose: torch.Tensor,
        wh: torch.Tensor,
        map_tokens: torch.Tensor,
        map_coordinates: torch.Tensor,
        image_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = pose.shape[0]
        x0_pose = self.encode_pose(pose, wh).unsqueeze(1).expand(
            -1, self.train_particles, -1
        )
        timesteps = torch.randint(
            0,
            self.num_train_steps,
            (batch_size, self.train_particles),
            device=pose.device,
        )
        noise = torch.randn_like(x0_pose)
        noisy_pose = self.q_sample(x0_pose, timesteps, noise)
        predicted_noise = self.denoiser(
            noisy_pose,
            timesteps,
            map_tokens,
            map_coordinates,
            image_global,
            wh,
            self.map_res,
        )

        xy_noise_loss = F.mse_loss(predicted_noise[..., :2], noise[..., :2])
        theta_noise_loss = F.mse_loss(predicted_noise[..., 2:4], noise[..., 2:4])
        noise_loss = xy_noise_loss + self.theta_loss_weight * theta_noise_loss

        return noise_loss, {
            "xy_noise_loss": xy_noise_loss,
            "theta_noise_loss": theta_noise_loss,
        }

    @torch.no_grad()
    def sample_from_context(
        self,
        map_tokens: torch.Tensor,
        map_coordinates: torch.Tensor,
        image_global: torch.Tensor,
        wh: torch.Tensor,
        num_particles: int = None,
        num_steps: int = None,
        cache_map_kv: bool = True,
    ) -> torch.Tensor:
        num_particles = int(num_particles or self.val_particles)
        num_steps = int(num_steps or self.sample_steps)
        batch_size = map_tokens.shape[0]
        state = torch.randn(
            batch_size,
            num_particles,
            4,
            device=map_tokens.device,
            dtype=map_tokens.dtype,
        )

        timestep_values = torch.linspace(
            self.num_train_steps - 1,
            0,
            num_steps,
            device=map_tokens.device,
        ).long()
        timestep_values = torch.unique_consecutive(timestep_values)
        map_kv_cache = (
            self.denoiser.build_map_kv_cache(map_tokens)
            if cache_map_kv
            else None
        )
        for step_idx, timestep_value in enumerate(timestep_values):
            timesteps = torch.full(
                (batch_size, num_particles),
                int(timestep_value.item()),
                device=map_tokens.device,
                dtype=torch.long,
            )
            predicted_noise = self.denoiser(
                state,
                timesteps,
                map_tokens,
                map_coordinates,
                image_global,
                wh,
                self.map_res,
                map_kv_cache=map_kv_cache,
            )
            x0_pose = self.predict_x0_pose(state, timesteps, predicted_noise)
            if step_idx == len(timestep_values) - 1:
                state = x0_pose
                break

            previous_timestep = timestep_values[step_idx + 1]
            previous_alpha = self.alpha_cumprod[previous_timestep]
            state = (
                previous_alpha.sqrt() * x0_pose
                + (1.0 - previous_alpha).sqrt() * predicted_noise
            )

        return self.decode_pose(state, wh)

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        checkpoint_hparams = checkpoint.get("hyper_parameters", {})
        loaded_convention = checkpoint_hparams.get(
            "diffusion_coordinate_convention",
            LEGACY_COORDINATE_CONVENTION,
        )
        if loaded_convention != self.coordinate_convention:
            raise RuntimeError(
                "This checkpoint uses an incompatible pose/map coordinate "
                "convention. Metric cell-center models must be trained from "
                "scratch (expected diffusion_coordinate_convention="
                f"{self.coordinate_convention!r}, got {loaded_convention!r})."
            )

    def select_pose_mode(
        self,
        pose_samples: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_delta_m = (
            pose_samples[:, :, None, :2] - pose_samples[:, None, :, :2]
        ) * self.map_res
        spatial_distance_sq = spatial_delta_m.square().sum(dim=-1)
        theta_delta = torch.remainder(
            pose_samples[:, :, None, 2]
            - pose_samples[:, None, :, 2]
            + math.pi,
            2 * math.pi,
        ) - math.pi
        distance = (
            spatial_distance_sq / max(self.mode_sigma_m**2, 1e-6)
            + theta_delta.square()
            / max(math.radians(self.mode_sigma_deg) ** 2, 1e-6)
        )
        density = torch.exp(-0.5 * distance).sum(dim=-1)
        mode_idx = density.argmax(dim=1)
        batch_idx = torch.arange(pose_samples.shape[0], device=pose_samples.device)
        return pose_samples[batch_idx, mode_idx], density

    @staticmethod
    def angular_error(
        pred_theta: torch.Tensor,
        gt_theta: torch.Tensor,
    ) -> torch.Tensor:
        return torch.abs(
            torch.remainder(pred_theta - gt_theta + math.pi, 2 * math.pi) - math.pi
        )

    def sample_metrics(
        self,
        pose_samples: torch.Tensor,
        selected_pose: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        selected_xy_error = torch.linalg.norm(
            selected_pose[:, :2] - target_pose[:, :2],
            dim=-1,
        ) * self.map_res
        selected_theta_error = self.angular_error(
            selected_pose[:, 2],
            target_pose[:, 2],
        )
        metrics = {
            "1m_recall": (selected_xy_error <= 1.0).float().mean(),
            "0.5m_recall": (selected_xy_error <= 0.5).float().mean(),
            "0.1m_recall": (selected_xy_error <= 0.1).float().mean(),
            "1m_30deg_recall": (
                (selected_xy_error <= 1.0)
                & (selected_theta_error <= math.radians(30.0))
            ).float().mean(),
            "mean_xy_err_m": selected_xy_error.mean(),
            "mean_theta_err_deg": torch.rad2deg(selected_theta_error).mean(),
        }

        all_xy_error = torch.linalg.norm(
            pose_samples[..., :2] - target_pose[:, None, :2],
            dim=-1,
        ) * self.map_res
        all_theta_error = self.angular_error(
            pose_samples[..., 2],
            target_pose[:, None, 2],
        )
        for count in (8, 32, 64):
            actual_count = min(count, pose_samples.shape[1])
            subset_xy = all_xy_error[:, :actual_count]
            subset_theta = all_theta_error[:, :actual_count]
            metrics[f"best_of_{count}_1m_recall"] = (
                subset_xy.min(dim=1).values <= 1.0
            ).float().mean()
            metrics[f"best_of_{count}_1m_30deg_recall"] = (
                ((subset_xy <= 1.0) & (subset_theta <= math.radians(30.0)))
                .any(dim=1)
                .float()
                .mean()
            )
        return metrics

    def _shared_step(self, batch, stage: str):
        obs_img, pose, _ray, floorplan_img, wh, _local_map, _neg_local_map, _neg_pose = batch
        map_tokens, map_coordinates, image_global = self.condition_encoder(
            obs_img,
            floorplan_img,
        )
        loss, loss_parts = self.diffusion_loss(
            pose,
            wh,
            map_tokens,
            map_coordinates,
            image_global,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {stage} diffusion loss: {loss.detach().item()}"
            )
        batch_size = obs_img.shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        for name, value in loss_parts.items():
            self.log(
                f"{stage}_{name}",
                value,
                prog_bar=False,
                batch_size=batch_size,
            )

        if stage == "val":
            pose_samples = self.sample_from_context(
                map_tokens,
                map_coordinates,
                image_global,
                wh,
            )
            selected_pose, density = self.select_pose_mode(pose_samples)
            metrics = self.sample_metrics(pose_samples, selected_pose, pose)
            for name, value in metrics.items():
                self.log(
                    f"val_{name}",
                    value,
                    prog_bar=name in ("1m_recall", "best_of_64_1m_recall"),
                    batch_size=batch_size,
                )
            self.log(
                "val_mode_peak_density",
                density.max(dim=1).values.mean(),
                prog_bar=False,
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.config.get("lr", 1e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(
                self.config.get(
                    "lr_t_max_epochs",
                    self.config.get("epochs", 30),
                )
            ),
            eta_min=float(self.config.get("min_lr", 1e-5)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
