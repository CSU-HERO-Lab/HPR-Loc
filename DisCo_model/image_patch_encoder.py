import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from DisCo_model.depth_anything_v2.dinov2 import DINOv2


class ImagePatchEncoder(nn.Module):
    def __init__(
        self,
        encoder="vits",
        feature_dim=128,
        target_size=(6, 40),
        intermediate_layer_idx=11,
        checkpoint_path=None,
        freeze_backbone=True,
        use_cls_token=False,
    ):
        super().__init__()

        if checkpoint_path is None:
            raise ValueError("ImagePatchEncoder requires a DINOv2 checkpoint path.")

        self.feature_dim = feature_dim
        self.target_size = tuple(target_size)
        self.intermediate_layer_idx = intermediate_layer_idx
        self.patch_size = 14
        self.use_cls_token = use_cls_token

        params = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.backbone = DINOv2(model_name=encoder)

        pretrained_dict = OrderedDict()
        for key, value in params.items():
            if key.startswith("pretrained."):
                pretrained_dict[key[11:]] = value

        self.backbone.load_state_dict(pretrained_dict, strict=True)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.proj = nn.Conv2d(384, feature_dim, kernel_size=1)
        if self.use_cls_token:
            self.cls_proj = nn.Linear(384, feature_dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, obs_img):
        bsz, _, height, width = obs_img.shape

        pad_height = int(math.ceil(height / self.patch_size) * self.patch_size)
        pad_width = int(math.ceil(width / self.patch_size) * self.patch_size)
        img_padded = F.pad(obs_img, (0, pad_width - width, 0, pad_height - height))

        image_features = self.backbone.get_intermediate_layers(
            img_padded,
            [self.intermediate_layer_idx],
            return_class_token=self.use_cls_token,
        )[0]
        if self.use_cls_token:
            patch_tokens, cls_token = image_features
        else:
            patch_tokens = image_features
            cls_token = None

        grid_h = pad_height // self.patch_size
        grid_w = pad_width // self.patch_size

        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(bsz, 384, grid_h, grid_w)
        patch_tokens = F.interpolate(
            patch_tokens,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        patch_tokens = self.proj(patch_tokens)

        _, _, feat_h, feat_w = patch_tokens.shape
        tokens = patch_tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self._build_2d_positional_encoding(
            feat_h, feat_w, bsz, tokens.device, tokens.dtype
        )

        if self.use_cls_token:
            return tokens, self.cls_proj(cls_token)

        return tokens

    def _build_2d_positional_encoding(self, feat_h, feat_w, batch_size, device, dtype):
        pos_x = torch.linspace(-0.5, 0.5, feat_w, device=device, dtype=dtype)
        pos_y = torch.linspace(-0.5, 0.5, feat_h, device=device, dtype=dtype)
        pos_grid_y, pos_grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        pos_grid = torch.stack((pos_grid_x, pos_grid_y), dim=-1).view(1, -1, 2)
        pos_enc = self.pos_mlp(pos_grid)
        return pos_enc.expand(batch_size, -1, -1)
