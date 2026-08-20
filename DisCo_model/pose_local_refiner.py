import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torchvision import transforms

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.floorplan_encoder import ResNetFloorplanEncoder
from DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def pose_delta_to_local_m(
    target_pose: torch.Tensor,
    candidate_pose: torch.Tensor,
    map_res: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    delta_px = target_pose[..., :2] - candidate_pose[..., :2]
    cos_theta = torch.cos(candidate_pose[..., 2])
    sin_theta = torch.sin(candidate_pose[..., 2])
    local_x_m = (cos_theta * delta_px[..., 0] + sin_theta * delta_px[..., 1]) * map_res
    local_y_m = (-sin_theta * delta_px[..., 0] + cos_theta * delta_px[..., 1]) * map_res
    delta_theta = wrap_to_pi(target_pose[..., 2] - candidate_pose[..., 2])
    return torch.stack([local_x_m, local_y_m], dim=-1), delta_theta


def apply_local_delta_to_pose(
    candidate_pose: torch.Tensor,
    delta_xy_m: torch.Tensor,
    delta_theta: torch.Tensor,
    map_res: float,
    wh: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cos_theta = torch.cos(candidate_pose[..., 2])
    sin_theta = torch.sin(candidate_pose[..., 2])
    dx_px = (cos_theta * delta_xy_m[..., 0] - sin_theta * delta_xy_m[..., 1]) / map_res
    dy_px = (sin_theta * delta_xy_m[..., 0] + cos_theta * delta_xy_m[..., 1]) / map_res
    xy = candidate_pose[..., :2] + torch.stack([dx_px, dy_px], dim=-1)
    if wh is not None:
        max_xy = wh.to(device=xy.device, dtype=xy.dtype) - 1.0
        xy = torch.minimum(torch.maximum(xy, torch.zeros_like(xy)), max_xy.clamp_min(0.0))
    theta = torch.remainder(candidate_pose[..., 2] + delta_theta, 2.0 * math.pi)
    return torch.cat([xy, theta.unsqueeze(-1)], dim=-1)


def crop_local_map_np(
    map_img: np.ndarray,
    pose: np.ndarray,
    crop_size_meters: float,
    map_res: float,
    output_size: int,
    oriented: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
    resize_interpolation: Optional[int] = None,
    border_value=255,
) -> np.ndarray:
    if resize_interpolation is None:
        resize_interpolation = cv2.INTER_AREA
    x, y, theta = [float(v) for v in pose[:3]]
    crop_size_px = max(1, int(round(crop_size_meters / map_res)))
    pad = crop_size_px
    map_padded = cv2.copyMakeBorder(
        map_img,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )

    center = (x + pad, y + pad)
    angle_deg = np.degrees(theta) + 90.0 if oriented else 0.0
    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot_matrix[0, 2] += (crop_size_px / 2.0) - center[0]
    rot_matrix[1, 2] += (crop_size_px / 2.0) - center[1]

    local_map = cv2.warpAffine(
        map_padded,
        rot_matrix,
        (crop_size_px, crop_size_px),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    if crop_size_px != output_size:
        local_map = cv2.resize(
            local_map,
            (output_size, output_size),
            interpolation=resize_interpolation,
        )
    return local_map


def load_refiner_map_np(dataset: DisCo_Dataset, floorplan_path: str) -> np.ndarray:
    representation = dataset.dataset_cfg.get(
        "refiner_floorplan_representation",
        dataset.floorplan_representation,
    )
    is_semantic_floorplan = (
        dataset.dataset_type in ("semrayloc", "clear_semrayloc")
        and Path(floorplan_path).name == "floorplan_semantic.png"
    )
    if representation == "semantic_onehot":
        with Image.open(floorplan_path) as map_img:
            rgb = np.asarray(map_img.convert("RGB"), dtype=np.uint8)
        return dataset._build_semantic_onehot_labels(rgb)

    if is_semantic_floorplan and representation != "gray":
        raise ValueError(
            "Semantic floorplans must use the semantic_onehot representation."
        )

    raw_map = cv2.imread(floorplan_path, cv2.IMREAD_GRAYSCALE)
    if raw_map is None:
        raise FileNotFoundError(f"Failed to load floorplan {floorplan_path}")
    return raw_map


def crop_to_refiner_tensor(
    refiner_map: np.ndarray,
    pose: np.ndarray,
    crop_size_meters: float,
    map_res: float,
    output_size: int,
    representation: str,
    oriented: bool = True,
) -> torch.Tensor:
    if representation == "semantic_onehot":
        local_labels = crop_local_map_np(
            refiner_map,
            pose,
            crop_size_meters=crop_size_meters,
            map_res=map_res,
            output_size=output_size,
            oriented=oriented,
            interpolation=cv2.INTER_NEAREST,
            resize_interpolation=cv2.INTER_NEAREST,
            border_value=4,
        ).astype(np.int64)
        local_labels = np.clip(local_labels, 0, 4)
        onehot = np.eye(5, dtype=np.float32)[local_labels]
        return torch.from_numpy(onehot).permute(2, 0, 1).contiguous()

    local_map_np = crop_local_map_np(
        refiner_map,
        pose,
        crop_size_meters=crop_size_meters,
        map_res=map_res,
        output_size=output_size,
        oriented=oriented,
    )
    return torch.from_numpy(local_map_np).float().unsqueeze(0) / 255.0


class PoseRefinerDataset(Dataset):
    def __init__(
        self,
        dataset_cfg: dict,
        split: str,
        floorplan_img_size: Tuple[int, int],
        crop_size_meters: float,
        crop_output_size: int,
        max_delta_m: float,
        max_delta_theta_deg: float,
        score_sigma_m: float,
        score_sigma_deg: float,
        deterministic: bool = False,
        seed: int = 0,
    ):
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.map_res = float(dataset_cfg.get("map_res", 0.02))
        self.crop_size_meters = float(crop_size_meters)
        self.crop_output_size = int(crop_output_size)
        self.max_delta_m = float(max_delta_m)
        self.max_delta_theta = math.radians(float(max_delta_theta_deg))
        self.score_sigma_m = float(score_sigma_m)
        self.score_sigma_theta = math.radians(float(score_sigma_deg))
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.refiner_floorplan_representation = dataset_cfg.get(
            "refiner_floorplan_representation",
            dataset_cfg.get("floorplan_representation", "rgb"),
        )
        self.refiner_oriented_crop = bool(dataset_cfg.get("refiner_oriented_crop", True))
        self.base_dataset = DisCo_Dataset(
            data_folder=dataset_cfg["data_folder"],
            data_splits_path=dataset_cfg["data_splits"],
            split=split,
            floorplan_img_size=floorplan_img_size,
            pose_aug_params={"enable": False},
            dataset_cfg=dataset_cfg,
        )
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.base_dataset.data)

    def _rng(self, index: int):
        if self.deterministic:
            return np.random.default_rng(self.seed + index)
        # Lightning seeds NumPy independently in every data-loader worker.
        return np.random

    def _sample_rotation(self, rng) -> int:
        if self.split != "train" or not self.base_dataset.map_pose_rot_aug_enable:
            return 0
        if rng.random() >= self.base_dataset.map_pose_rot_aug_p:
            return 0
        return int(rng.choice(self.base_dataset.map_pose_rot_aug_angles))

    def _sample_candidate(self, pose: np.ndarray, wh: Tuple[int, int], rng) -> np.ndarray:
        mode = float(rng.random())
        if mode < 0.5:
            trans_sigma_m = 0.3
            theta_sigma = math.radians(10.0)
            offset_m = rng.normal(0.0, trans_sigma_m, size=2)
            theta_delta = float(rng.normal(0.0, theta_sigma))
        elif mode < 0.8:
            trans_sigma_m = 0.8
            theta_sigma = math.radians(25.0)
            offset_m = rng.normal(0.0, trans_sigma_m, size=2)
            theta_delta = float(rng.normal(0.0, theta_sigma))
        else:
            radius = self.max_delta_m * math.sqrt(float(rng.random()))
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            offset_m = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * radius
            theta_delta = float(rng.uniform(-self.max_delta_theta, self.max_delta_theta))

        norm = float(np.linalg.norm(offset_m))
        if norm > self.max_delta_m:
            offset_m = offset_m / max(norm, 1e-6) * self.max_delta_m
        theta_delta = float(np.clip(theta_delta, -self.max_delta_theta, self.max_delta_theta))

        candidate = pose.astype(np.float32).copy()
        candidate[:2] = candidate[:2] + offset_m.astype(np.float32) / self.map_res
        candidate[0] = np.clip(candidate[0], 0.0, max(float(wh[0] - 1), 0.0))
        candidate[1] = np.clip(candidate[1], 0.0, max(float(wh[1] - 1), 0.0))
        candidate[2] = np.mod(candidate[2] + theta_delta, 2.0 * math.pi)
        return candidate

    def __getitem__(self, index: int):
        rng = self._rng(index)
        data = self.base_dataset.data[index]
        obs_img = self.image_transform(Image.open(data["rgb_image"]).convert("RGB"))

        with Image.open(data["floorplan_image"]) as map_img:
            width, height = map_img.size
        pose_np = np.asarray(data["pose"], dtype=np.float32).copy()
        rotation_k = self._sample_rotation(rng)
        if rotation_k:
            pose_np, width, height = self.base_dataset._rotate_pose_wh_90(
                pose_np,
                width,
                height,
                rotation_k,
            )

        refiner_map = load_refiner_map_np(self.base_dataset, data["floorplan_image"])
        refiner_map = self.base_dataset._rotate_array_90(refiner_map, rotation_k)

        candidate_np = self._sample_candidate(pose_np, (width, height), rng)
        local_map = crop_to_refiner_tensor(
            refiner_map,
            candidate_np,
            crop_size_meters=self.crop_size_meters,
            map_res=self.map_res,
            output_size=self.crop_output_size,
            representation=self.refiner_floorplan_representation,
            oriented=self.refiner_oriented_crop,
        )

        pose = torch.from_numpy(pose_np.astype(np.float32))
        candidate_pose = torch.from_numpy(candidate_np.astype(np.float32))
        delta_xy_m, delta_theta = pose_delta_to_local_m(
            pose,
            candidate_pose,
            map_res=self.map_res,
        )
        candidate_xy_error_m = torch.linalg.norm(
            (candidate_pose[:2] - pose[:2]) * self.map_res,
            dim=-1,
        )
        candidate_theta_error = torch.abs(wrap_to_pi(candidate_pose[2] - pose[2]))
        score_target = torch.exp(
            -0.5 * (candidate_xy_error_m / max(self.score_sigma_m, 1e-6)).square()
            -0.5 * (candidate_theta_error / max(self.score_sigma_theta, 1e-6)).square()
        )

        return {
            "obs_img": obs_img.float(),
            "local_map": local_map.float(),
            "pose": pose.float(),
            "candidate_pose": candidate_pose.float(),
            "wh": torch.tensor([width, height], dtype=torch.float32),
            "target_delta_xy_m": delta_xy_m.float(),
            "target_delta_theta": delta_theta.float(),
            "score_target": score_target.float(),
        }


class PoseLocalRefiner(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.feature_dim = int(config.get("refiner_feature_dim", 128))
        self.max_delta_m = float(config.get("refiner_max_delta_m", 1.5))
        self.max_delta_theta = math.radians(float(config.get("refiner_max_delta_theta_deg", 45.0)))
        self.map_res = float(config["datasets"].get("map_res", 0.02))
        self.dropout = nn.Dropout(float(config.get("refiner_dropout", 0.1)))
        self.refiner_map_input_mode = config.get(
            "refiner_map_input_mode",
            config["diffusion_map_input_mode"],
        )

        diffusion = PoseQueryDiffusionLocalizer(config)
        checkpoint = torch.load(
            config["baseline_checkpoint_path"],
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        image_state_dict = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("condition_encoder.image_encoder.")
            or key.startswith("condition_encoder.image_norm.")
            or key.startswith("condition_encoder.image_mixer.")
        }
        diffusion.load_state_dict(image_state_dict, strict=False)
        self.image_encoder = diffusion.condition_encoder.image_encoder
        self.image_norm = diffusion.condition_encoder.image_norm
        self.image_mixer = diffusion.condition_encoder.image_mixer
        for module in (self.image_encoder, self.image_norm, self.image_mixer):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        self.local_map_encoder = ResNetFloorplanEncoder(
            feature_dim=self.feature_dim,
            input_mode=self.refiner_map_input_mode,
            context_blocks=int(config.get("refiner_map_context_blocks", 2)),
            pretrained=False,
        )
        self.map_pos_mlp = nn.Sequential(
            nn.Linear(2, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.map_norm = nn.LayerNorm(self.feature_dim)
        self.pose_mlp = nn.Sequential(
            nn.Linear(4, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.image_global_projection = nn.Linear(self.feature_dim, self.feature_dim)
        self.query_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        nn.init.normal_(self.query_token, std=0.02)

        num_heads = int(config.get("refiner_num_heads", config.get("diffusion_num_heads", 4)))
        self.image_attn = nn.MultiheadAttention(
            self.feature_dim,
            num_heads,
            dropout=float(config.get("refiner_dropout", 0.1)),
            batch_first=True,
        )
        self.image_attn_norm = nn.LayerNorm(self.feature_dim)
        self.map_attn = nn.MultiheadAttention(
            self.feature_dim,
            num_heads,
            dropout=float(config.get("refiner_dropout", 0.1)),
            batch_first=True,
        )
        self.map_attn_norm = nn.LayerNorm(self.feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 4),
            nn.GELU(),
            nn.Dropout(float(config.get("refiner_dropout", 0.1))),
            nn.Linear(self.feature_dim * 4, self.feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(self.feature_dim)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 3),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
        )

    @staticmethod
    def build_map_coordinates(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pos_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        pos_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in (self.image_encoder, self.image_norm, self.image_mixer):
            module.eval()
        if hasattr(self.image_encoder, "backbone"):
            self.image_encoder.backbone.eval()
        return self

    def encode_image(self, obs_img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_tokens = self.image_encoder(obs_img)
            image_tokens = self.image_mixer(self.image_norm(image_tokens))
        return image_tokens.detach()

    def encode_local_map(self, local_map: torch.Tensor) -> torch.Tensor:
        map_features = self.local_map_encoder(local_map)
        _, _, height, width = map_features.shape
        map_tokens = map_features.flatten(2).transpose(1, 2)
        coordinates = self.build_map_coordinates(
            height,
            width,
            map_tokens.device,
            map_tokens.dtype,
        )
        map_tokens = map_tokens + self.map_pos_mlp(coordinates).unsqueeze(0)
        return self.map_norm(map_tokens)

    def encode_candidate_pose(
        self,
        candidate_pose: torch.Tensor,
        wh: torch.Tensor,
    ) -> torch.Tensor:
        xy = candidate_pose[:, :2] / wh.clamp_min(1.0)
        xy = xy * 2.0 - 1.0
        angle = torch.stack(
            [torch.sin(candidate_pose[:, 2]), torch.cos(candidate_pose[:, 2])],
            dim=-1,
        )
        return torch.cat([xy, angle], dim=-1)

    def forward(
        self,
        obs_img: torch.Tensor,
        local_map: torch.Tensor,
        candidate_pose: torch.Tensor,
        wh: torch.Tensor,
        image_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if image_tokens is None:
            image_tokens = self.encode_image(obs_img)
        map_tokens = self.encode_local_map(local_map)
        image_global = image_tokens.mean(dim=1)
        pose_features = self.encode_candidate_pose(candidate_pose, wh)

        query = self.query_token.expand(obs_img.shape[0], -1, -1)
        query = query + self.pose_mlp(pose_features).unsqueeze(1)
        query = query + self.image_global_projection(image_global).unsqueeze(1)

        image_attended, _ = self.image_attn(
            query=query,
            key=image_tokens,
            value=image_tokens,
            need_weights=False,
        )
        query = self.image_attn_norm(query + self.dropout(image_attended))
        map_attended, _ = self.map_attn(
            query=query,
            key=map_tokens,
            value=map_tokens,
            need_weights=False,
        )
        query = self.map_attn_norm(query + self.dropout(map_attended))
        query = self.ffn_norm(query + self.dropout(self.ffn(query)))
        query = query.squeeze(1)

        raw_delta = self.delta_head(query)
        delta_xy_m = torch.tanh(raw_delta[:, :2]) * self.max_delta_m
        delta_theta = torch.tanh(raw_delta[:, 2]) * self.max_delta_theta
        score_logit = self.score_head(query).squeeze(-1)
        return {
            "delta_xy_m": delta_xy_m,
            "delta_theta": delta_theta,
            "score_logit": score_logit,
        }


class DensePoseLocalRefiner(PoseLocalRefiner):
    """Image-conditioned dense likelihood over a candidate-centered map crop."""

    def __init__(self, config: dict):
        super().__init__(config)
        for module in (
            self.image_attn,
            self.image_attn_norm,
            self.map_attn,
            self.map_attn_norm,
            self.ffn,
            self.ffn_norm,
            self.delta_head,
            self.score_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.query_token.requires_grad = False
        num_heads = int(config.get("refiner_num_heads", config.get("diffusion_num_heads", 4)))
        dropout = float(config.get("refiner_dropout", 0.1))
        self.crop_size_meters = float(config.get("refiner_crop_size_meters", 5.0))
        self.dense_temperature = float(config.get("refiner_dense_temperature", 1.0))
        self.dense_image_attn = nn.MultiheadAttention(
            self.feature_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dense_image_attn_norm = nn.LayerNorm(self.feature_dim)
        self.dense_ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
        )
        self.dense_ffn_norm = nn.LayerNorm(self.feature_dim)
        self.heatmap_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, 1),
        )
        self.theta_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, 2),
        )
        self.dense_score_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, 1),
        )

    @staticmethod
    def build_dense_coordinates(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pos_x = ((torch.arange(width, device=device, dtype=dtype) + 0.5) / width) * 2.0 - 1.0
        pos_y = ((torch.arange(height, device=device, dtype=dtype) + 0.5) / height) * 2.0 - 1.0
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

    @staticmethod
    def coordinates_to_local_m(
        coordinates: torch.Tensor,
        crop_size_meters: float,
    ) -> torch.Tensor:
        # Oriented crops place candidate-forward toward image-up and left toward image-right.
        half_size = crop_size_meters * 0.5
        return torch.stack(
            [-coordinates[..., 1], coordinates[..., 0]], dim=-1
        ) * half_size

    def forward(
        self,
        obs_img: torch.Tensor,
        local_map: torch.Tensor,
        candidate_pose: torch.Tensor,
        wh: torch.Tensor,
        image_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if image_tokens is None:
            image_tokens = self.encode_image(obs_img)
        map_tokens = self.encode_local_map(local_map)
        coordinates = self.build_dense_coordinates(
            int(round(math.sqrt(map_tokens.shape[1]))),
            int(round(math.sqrt(map_tokens.shape[1]))),
            map_tokens.device,
            map_tokens.dtype,
        )
        if coordinates.shape[0] != map_tokens.shape[1]:
            raise ValueError("Dense refiner expects a square local-map token grid.")

        image_global = image_tokens.mean(dim=1)
        pose_features = self.encode_candidate_pose(candidate_pose, wh)
        map_tokens = map_tokens + self.image_global_projection(image_global).unsqueeze(1)
        map_tokens = map_tokens + self.pose_mlp(pose_features).unsqueeze(1)
        attended, _ = self.dense_image_attn(
            query=map_tokens,
            key=image_tokens,
            value=image_tokens,
            need_weights=False,
        )
        fused_tokens = self.dense_image_attn_norm(
            map_tokens + self.dropout(attended)
        )
        fused_tokens = self.dense_ffn_norm(
            fused_tokens + self.dropout(self.dense_ffn(fused_tokens))
        )

        heatmap_logits = self.heatmap_head(fused_tokens).squeeze(-1)
        heatmap_prob = F.softmax(
            heatmap_logits / max(self.dense_temperature, 1e-6), dim=-1
        )
        local_coordinates_m = self.coordinates_to_local_m(
            coordinates, self.crop_size_meters
        )
        delta_xy_m = torch.einsum("bn,nd->bd", heatmap_prob, local_coordinates_m)

        theta_vectors = self.theta_head(fused_tokens)
        theta_vector = torch.einsum("bn,bnd->bd", heatmap_prob, theta_vectors)
        delta_theta = torch.atan2(theta_vector[:, 0], theta_vector[:, 1]).clamp(
            -self.max_delta_theta, self.max_delta_theta
        )
        pooled_features = torch.einsum("bn,bnd->bd", heatmap_prob, fused_tokens)
        score_logit = self.dense_score_head(pooled_features).squeeze(-1)
        return {
            "delta_xy_m": delta_xy_m,
            "delta_theta": delta_theta,
            "score_logit": score_logit,
            "heatmap_logits": heatmap_logits,
            "local_coordinates_m": local_coordinates_m,
        }


class PoseLocalRefinerLightning(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        self.map_res = float(config["datasets"].get("map_res", 0.02))
        self.delta_xy_weight = float(config.get("refiner_delta_xy_loss_weight", 1.0))
        self.delta_theta_weight = float(config.get("refiner_delta_theta_loss_weight", 1.0))
        self.score_weight = float(config.get("refiner_score_loss_weight", 0.25))
        self.dense_heatmap_weight = float(config.get("refiner_dense_heatmap_loss_weight", 1.0))
        self.dense_heatmap_sigma_m = float(config.get("refiner_dense_heatmap_sigma_m", 0.2))
        refiner_arch = config.get("refiner_arch", "dense_heatmap")
        if refiner_arch == "query_regression":
            self.refiner = PoseLocalRefiner(config)
        elif refiner_arch == "dense_heatmap":
            self.refiner = DensePoseLocalRefiner(config)
        else:
            raise ValueError(f"Unsupported refiner_arch: {refiner_arch}")

    def train(self, mode: bool = True):
        super().train(mode)
        self.refiner.train(mode)
        return self

    def forward(
        self,
        obs_img: torch.Tensor,
        local_map: torch.Tensor,
        candidate_pose: torch.Tensor,
        wh: torch.Tensor,
        image_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.refiner(obs_img, local_map, candidate_pose, wh, image_tokens)

    @staticmethod
    def _xy_error_m(pred_pose: torch.Tensor, target_pose: torch.Tensor, map_res: float):
        return torch.linalg.norm(pred_pose[:, :2] - target_pose[:, :2], dim=-1) * map_res

    @staticmethod
    def _theta_error(pred_pose: torch.Tensor, target_pose: torch.Tensor):
        return torch.abs(wrap_to_pi(pred_pose[:, 2] - target_pose[:, 2]))

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str):
        outputs = self(
            batch["obs_img"],
            batch["local_map"],
            batch["candidate_pose"],
            batch["wh"],
        )
        xy_loss = F.smooth_l1_loss(outputs["delta_xy_m"], batch["target_delta_xy_m"])
        theta_loss = (
            1.0 - torch.cos(outputs["delta_theta"] - batch["target_delta_theta"])
        ).mean()
        score_loss = F.binary_cross_entropy_with_logits(
            outputs["score_logit"],
            batch["score_target"],
        )
        dense_heatmap_loss = outputs["delta_xy_m"].new_zeros(())
        if "heatmap_logits" in outputs:
            squared_distance = (
                outputs["local_coordinates_m"].unsqueeze(0)
                - batch["target_delta_xy_m"].unsqueeze(1)
            ).square().sum(dim=-1)
            target = torch.exp(
                -0.5
                * squared_distance
                / max(self.dense_heatmap_sigma_m, 1e-6) ** 2
            )
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            dense_heatmap_loss = -(
                target * F.log_softmax(outputs["heatmap_logits"], dim=-1)
            ).sum(dim=-1).mean()
        loss = (
            self.delta_xy_weight * xy_loss
            + self.delta_theta_weight * theta_loss
            + self.score_weight * score_loss
            + self.dense_heatmap_weight * dense_heatmap_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {stage} refiner loss: {loss.detach().item()}"
            )

        refined_pose = apply_local_delta_to_pose(
            batch["candidate_pose"],
            outputs["delta_xy_m"],
            outputs["delta_theta"],
            map_res=self.map_res,
            wh=batch["wh"],
        )
        candidate_xy_error = self._xy_error_m(
            batch["candidate_pose"],
            batch["pose"],
            self.map_res,
        )
        refined_xy_error = self._xy_error_m(refined_pose, batch["pose"], self.map_res)
        refined_theta_error = self._theta_error(refined_pose, batch["pose"])
        batch_size = batch["obs_img"].shape[0]

        logs = {
            f"{stage}_loss": loss,
            f"{stage}_delta_xy_loss": xy_loss,
            f"{stage}_delta_theta_loss": theta_loss,
            f"{stage}_score_loss": score_loss,
            f"{stage}_dense_heatmap_loss": dense_heatmap_loss,
            f"{stage}_candidate_0.5m_recall": (candidate_xy_error <= 0.5).float().mean(),
            f"{stage}_candidate_1m_recall": (candidate_xy_error <= 1.0).float().mean(),
            f"{stage}_refined_0.1m_recall": (refined_xy_error <= 0.1).float().mean(),
            f"{stage}_refined_0.5m_recall": (refined_xy_error <= 0.5).float().mean(),
            f"{stage}_refined_1m_recall": (refined_xy_error <= 1.0).float().mean(),
            f"{stage}_refined_1m_30deg_recall": (
                (refined_xy_error <= 1.0)
                & (refined_theta_error <= math.radians(30.0))
            ).float().mean(),
            f"{stage}_mean_xy_err_m": refined_xy_error.mean(),
        }
        for name, value in logs.items():
            self.log(
                name,
                value,
                prog_bar=name in (f"{stage}_loss", f"{stage}_refined_0.5m_recall"),
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=float(self.config.get("lr", 1e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(self.config.get("epochs", 30)),
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


def load_refiner_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
