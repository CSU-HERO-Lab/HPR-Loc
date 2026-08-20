import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from DisCo_model.image_patch_encoder import ImagePatchEncoder
from DisCo_model.map_encoder import MapEncoder


class DisCoLocModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.feature_dim = config.get("feature_dim", 128)
        self.num_heads = config.get("num_heads", 4)
        self.image_token_grid = tuple(config.get("image_token_grid", [6, 40]))
        self.image_self_attn_layers = config.get("image_self_attn_layers", 1)
        self.pairwise_chunk_size = config.get("pairwise_chunk_size", 16)
        self.hard_negative_mode = self._get_hard_negative_mode()
        self.use_cls_global_fusion = bool(config.get("use_cls_global_fusion", False))
        self.use_cls_query_token = bool(config.get("use_cls_query_token", False))
        self.use_image_cls_token = self.use_cls_global_fusion or self.use_cls_query_token

        self.image_encoder = ImagePatchEncoder(
            encoder=config.get("image_encoder", "vits"),
            feature_dim=self.feature_dim,
            target_size=self.image_token_grid,
            checkpoint_path=config.get(
                "dptv2_ckpt_path", "checkpoints/depth_anything_v2_vits.pth"
            ),
            freeze_backbone=config.get("freeze_image_backbone", True),
            use_cls_token=self.use_image_cls_token,
        )

        self.image_token_norm = nn.LayerNorm(self.feature_dim)
        self.image_self_attn = self._build_image_token_mixer()
        if self.use_image_cls_token:
            self.cls_token_norm = nn.LayerNorm(self.feature_dim)

        local_map_representation = config.get("datasets", {}).get(
            "local_map_representation",
            "semantic_onehot"
            if config.get("datasets", {}).get("floorplan_representation")
            == "semantic_onehot"
            else "gray",
        )
        input_channels = {"gray": 1, "semantic_onehot": 5}.get(
            local_map_representation
        )
        if input_channels is None:
            raise ValueError(
                "DisCo local_map_representation must be 'gray' or "
                "'semantic_onehot'."
            )
        self.map_encoder = MapEncoder(
            input_channels=input_channels,
            feature_dim=self.feature_dim,
        )
        self.map_pos_mlp = nn.Sequential(
            nn.Linear(2, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.map_token_norm = nn.LayerNorm(self.feature_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(self.feature_dim)

        self.token_score_head = nn.Linear(self.feature_dim, 1)
        if self.use_cls_global_fusion:
            self.cls_pair_fusion = nn.Sequential(
                nn.LayerNorm(self.feature_dim * 2),
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.GELU(),
            )
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _get_hard_negative_mode(self):
        dataset_cfg = self.config.get("datasets", {})
        hard_negative_cfg = dataset_cfg.get("hard_negative", {})
        if isinstance(hard_negative_cfg, dict):
            mode = hard_negative_cfg.get("mode", None)
        else:
            mode = hard_negative_cfg

        mode = dataset_cfg.get("hard_negative_mode", mode)
        mode = (mode or "mixed").lower()
        aliases = {
            "pos": "position",
            "trans": "position",
            "translation": "position",
            "ori": "orientation",
            "rot": "orientation",
            "rotation": "orientation",
            "off": "none",
            "false": "none",
            "disable": "none",
            "disabled": "none",
        }
        mode = aliases.get(mode, mode)
        if mode not in ("mixed", "position", "orientation", "none"):
            raise ValueError(
                "Unsupported hard negative mode "
                f"'{mode}'. Expected one of: mixed, position, orientation, none."
            )
        return mode

    def _build_image_token_mixer(self):
        if self.image_self_attn_layers <= 0:
            return nn.Identity()

        layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feature_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(
            layer,
            num_layers=self.image_self_attn_layers,
            norm=nn.LayerNorm(self.feature_dim),
        )

    def encode_image(self, obs_img):
        image_features = self.image_encoder(obs_img)
        if self.use_image_cls_token:
            img_tokens, cls_token = image_features
        else:
            img_tokens = image_features
            cls_token = None

        img_tokens = self.image_token_norm(img_tokens)
        img_tokens = self.image_self_attn(img_tokens)
        if cls_token is not None:
            cls_token = self.cls_token_norm(cls_token)
            if self.use_cls_query_token:
                img_tokens = torch.cat([cls_token.unsqueeze(1), img_tokens], dim=1)
            if not self.use_cls_global_fusion:
                return img_tokens
            return img_tokens, cls_token
        return img_tokens

    def encode_map(self, local_map):
        map_feat = self.map_encoder(local_map)
        bsz, channels, feat_h, feat_w = map_feat.shape
        map_tokens = map_feat.view(bsz, channels, -1).permute(0, 2, 1)
        map_tokens = map_tokens + self._build_map_positional_encoding(
            feat_h, feat_w, bsz, map_tokens.device, map_tokens.dtype
        )
        map_tokens = self.map_token_norm(map_tokens)
        return map_tokens

    def _build_map_positional_encoding(self, feat_h, feat_w, batch_size, device, dtype):
        pos_x = torch.linspace(-0.5, 0.5, feat_w, device=device, dtype=dtype)
        pos_y = torch.linspace(-0.5, 0.5, feat_h, device=device, dtype=dtype)
        pos_grid_y, pos_grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        pos_grid = torch.stack((pos_grid_x, pos_grid_y), dim=-1).view(1, -1, 2)
        pos_enc = self.map_pos_mlp(pos_grid)
        return pos_enc.expand(batch_size, -1, -1)

    def _split_image_features(self, image_features):
        if isinstance(image_features, tuple):
            return image_features
        return image_features, None

    def _score_encoded_pairs(
        self,
        img_tokens,
        map_tokens,
        cls_features=None,
        return_attn=False,
        return_token_attn=False,
    ):
        aligned_tokens, attn_weights = self.cross_attn(
            query=img_tokens,
            key=map_tokens,
            value=map_tokens,
            need_weights=return_attn or return_token_attn,
        )
        fused_tokens = self.cross_attn_norm(img_tokens + aligned_tokens)

        token_logits = self.token_score_head(fused_tokens).squeeze(-1)
        token_weights = torch.softmax(token_logits, dim=1)
        pooled = torch.sum(fused_tokens * token_weights.unsqueeze(-1), dim=1)
        if self.use_cls_global_fusion:
            if cls_features is None:
                raise ValueError("cls_features is required when use_cls_global_fusion=True.")
            pooled = self.cls_pair_fusion(torch.cat([pooled, cls_features], dim=-1))
        scores = self.score_head(pooled).squeeze(-1)

        if not (return_attn or return_token_attn):
            return scores

        map_attn = torch.einsum("bl,blm->bm", token_weights, attn_weights)
        if return_token_attn:
            return scores, map_attn, attn_weights
        return scores, map_attn

    def _pairwise_scores(self, image_features_all, map_tokens_all):
        img_tokens_all, cls_tokens_all = self._split_image_features(image_features_all)
        num_candidates = map_tokens_all.shape[0]
        chunk_size = min(self.pairwise_chunk_size, num_candidates)
        score_chunks = []

        for start in range(0, num_candidates, chunk_size):
            end = min(start + chunk_size, num_candidates)
            map_chunk = map_tokens_all[start:end]
            current_chunk = end - start

            query_tokens = (
                img_tokens_all.unsqueeze(1)
                .expand(-1, current_chunk, -1, -1)
                .reshape(-1, img_tokens_all.shape[1], img_tokens_all.shape[2])
            )
            map_tokens = (
                map_chunk.unsqueeze(0)
                .expand(img_tokens_all.shape[0], -1, -1, -1)
                .reshape(-1, map_chunk.shape[1], map_chunk.shape[2])
            )
            cls_features = None
            if cls_tokens_all is not None:
                cls_features = (
                    cls_tokens_all.unsqueeze(1)
                    .expand(-1, current_chunk, -1)
                    .reshape(-1, cls_tokens_all.shape[1])
                )

            chunk_scores = self._score_encoded_pairs(
                query_tokens, map_tokens, cls_features=cls_features
            )
            score_chunks.append(chunk_scores.view(img_tokens_all.shape[0], current_chunk))

        return torch.cat(score_chunks, dim=1)

    def forward(self, obs_img, local_map, return_attn=False):
        image_features = self.encode_image(obs_img)
        map_tokens = self.encode_map(local_map)
        img_tokens, cls_token = self._split_image_features(image_features)
        outputs = self._score_encoded_pairs(
            img_tokens, map_tokens, cls_features=cls_token, return_attn=return_attn
        )

        if return_attn:
            pair_scores, map_attn = outputs
            return image_features, pair_scores, map_attn

        return image_features, outputs

    def score_candidates(self, img_tokens, candidate_maps, return_attn=False):
        cls_token = None
        if isinstance(img_tokens, tuple):
            img_tokens, cls_token = img_tokens

        if img_tokens.dim() == 2:
            img_tokens = img_tokens.unsqueeze(0)
        if cls_token is not None and cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)

        num_candidates = candidate_maps.shape[0]
        map_tokens = self.encode_map(candidate_maps)

        if img_tokens.shape[0] == 1:
            img_tokens = img_tokens.expand(num_candidates, -1, -1)
            if cls_token is not None:
                cls_token = cls_token.expand(num_candidates, -1)
        elif img_tokens.shape[0] != num_candidates:
            raise ValueError(
                f"Image token batch ({img_tokens.shape[0]}) must be 1 or match the "
                f"candidate count ({num_candidates})."
            )
        elif cls_token is not None and cls_token.shape[0] != num_candidates:
            raise ValueError(
                f"CLS token batch ({cls_token.shape[0]}) must be 1 or match the "
                f"candidate count ({num_candidates})."
            )

        return self._score_encoded_pairs(
            img_tokens, map_tokens, cls_features=cls_token, return_attn=return_attn
        )

    def training_step(self, batch, batch_idx):
        obs_img, pose, ray, floorplan_img, wh, local_map, neg_local_map, neg_pose = batch
        batch_size = obs_img.shape[0]

        img_tokens_all = self.encode_image(obs_img)

        if self.hard_negative_mode == "none":
            map_all_input = local_map
        else:
            map_all_input = torch.cat([local_map, neg_local_map], dim=0)
        map_tokens_all = self.encode_map(map_all_input)

        logits_matrix = self._pairwise_scores(img_tokens_all, map_tokens_all)
        logit_scale = self.logit_scale.exp()
        logits_matrix = logits_matrix * logit_scale

        targets = torch.arange(batch_size, device=self.device).long()
        loss = F.cross_entropy(logits_matrix, targets)
        if self.trainer is not None and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", current_lr, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("scale", logit_scale)

        with torch.no_grad():
            pred = torch.argmax(logits_matrix, dim=1)
            correct = (pred == targets).float().sum()
            acc = correct / batch_size
            self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        obs_img, pose, _, floorplan_img, wh, local_map, _, _ = batch
        batch_size = obs_img.shape[0]

        img_tokens_all = self.encode_image(obs_img)
        map_tokens_all = self.encode_map(local_map)

        logits_matrix = self._pairwise_scores(img_tokens_all, map_tokens_all)
        logit_scale = self.logit_scale.exp()
        logits_matrix = logits_matrix * logit_scale

        targets = torch.arange(batch_size, device=self.device).long()
        loss_val = F.cross_entropy(logits_matrix, targets)

        self.log("val_loss", loss_val, prog_bar=True)

        with torch.no_grad():
            pred_indices = torch.argmax(logits_matrix, dim=1)
            correct = (pred_indices == targets).float().sum()
            acc_val = correct / batch_size
            self.log("val_acc", acc_val, prog_bar=True)

        return loss_val

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.get("lr", 1e-4))
        scheduler_name = self.config.get("lr_scheduler", "cosine")

        if scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.get("epochs", 30),
                eta_min=self.config.get("min_lr", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
