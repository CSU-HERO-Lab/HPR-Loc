#!/usr/bin/env python3
"""Evaluate the complete HPR-Loc diffusion, DisCo, and refiner pipeline."""

import argparse
import json
import math
import os
import random
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.pose_local_refiner import (
    PoseLocalRefinerLightning,
    apply_local_delta_to_pose,
    crop_to_refiner_tensor,
)
from DisCo_model.pose_query_diffusion import PoseQueryDiffusionLocalizer
from training.DisCo_lightning_module import DisCoLocModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Diffusion config.")
    parser.add_argument("--refiner-config", required=True)
    parser.add_argument("--diffusion-ckpt", required=True)
    parser.add_argument("--disco-ckpt", required=True)
    parser.add_argument("--refiner-ckpt", required=True)
    parser.add_argument(
        "--depth-ckpt", default="checkpoints/depth_anything_v2_vits.pth"
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", default="outputs/hpr_loc_test.json")
    parser.add_argument("--data-root")
    parser.add_argument("--split-yaml")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--num-modes", type=int, default=5)
    parser.add_argument("--nms-xy-m", type=float, default=1.0)
    parser.add_argument("--nms-theta-deg", type=float, default=30.0)
    parser.add_argument("--disco-crop-m", type=float, default=7.0)
    parser.add_argument("--disco-weight", type=float, default=0.9)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.num_modes < 1 or args.num_modes > args.num_particles:
        parser.error("--num-modes must be in [1, num-particles]")
    if not 0.0 <= args.disco_weight <= 1.0:
        parser.error("--disco-weight must lie in [0, 1]")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IndexedDataset(Dataset):
    def __init__(self, dataset, count):
        self.dataset = dataset
        self.count = min(len(dataset), count or len(dataset))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return index, self.dataset[index]


class FloorplanCache:
    def __init__(self, max_items=64):
        self.max_items = max_items
        self.cache = OrderedDict()

    def get(self, path, representation, dataset):
        key = (path, representation)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if representation == "semantic_onehot":
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            floorplan = dataset._build_semantic_onehot_labels(rgb)
        else:
            floorplan = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if floorplan is None:
                raise FileNotFoundError(f"Failed to read floorplan: {path}")
        self.cache[key] = floorplan
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return floorplan


class Metrics:
    def __init__(self):
        self.xy_errors = []
        self.theta_errors = []

    def update(self, poses, targets, map_res):
        xy = torch.linalg.norm((poses[:, :2] - targets[:, :2]) * map_res, dim=-1)
        theta = torch.abs(
            torch.remainder(poses[:, 2] - targets[:, 2] + math.pi, 2 * math.pi)
            - math.pi
        )
        self.xy_errors.extend(xy.detach().cpu().tolist())
        self.theta_errors.extend(theta.detach().cpu().tolist())

    def summarize(self):
        xy = torch.tensor(self.xy_errors)
        theta = torch.tensor(self.theta_errors)
        return {
            "samples": len(xy),
            "0.1m_recall": float((xy <= 0.1).float().mean()),
            "0.5m_recall": float((xy <= 0.5).float().mean()),
            "1m_recall": float((xy <= 1.0).float().mean()),
            "1m_30deg_recall": float(
                ((xy <= 1.0) & (theta <= math.radians(30))).float().mean()
            ),
            "mean_xy_err_m": float(xy.mean()),
            "median_xy_err_m": float(xy.median()),
            "mean_theta_err_deg": float(torch.rad2deg(theta).mean()),
        }


def extract_kde_modes(poses, density, count, map_res, xy_radius_m, theta_radius_deg):
    selected = []
    theta_radius = math.radians(theta_radius_deg)
    for index in torch.argsort(density, descending=True).tolist():
        candidate = poses[index]
        if selected:
            modes = poses[selected]
            xy = torch.linalg.norm((modes[:, :2] - candidate[:2]) * map_res, dim=-1)
            theta = torch.abs(
                torch.remainder(modes[:, 2] - candidate[2] + math.pi, 2 * math.pi)
                - math.pi
            )
            distance = (xy / xy_radius_m).square() + (
                theta / theta_radius
            ).square()
            if torch.any(distance <= 1.0):
                continue
        selected.append(index)
        if len(selected) == count:
            break
    return torch.tensor(selected, device=poses.device, dtype=torch.long)


def normalize_scores(scores):
    if scores.numel() <= 1:
        return torch.zeros_like(scores)
    return (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-6)


def fused_candidate_scores(disco_scores, density, disco_weight):
    return disco_weight * normalize_scores(disco_scores) + (
        1.0 - disco_weight
    ) * normalize_scores(torch.log(density.clamp_min(1e-8)))


def project_patch_tokens(image_encoder, patch_tokens, grid_h, grid_w):
    batch_size = patch_tokens.shape[0]
    features = patch_tokens.permute(0, 2, 1).reshape(
        batch_size, patch_tokens.shape[-1], grid_h, grid_w
    )
    features = F.interpolate(
        features, size=image_encoder.target_size, mode="bilinear", align_corners=False
    )
    features = image_encoder.proj(features)
    height, width = image_encoder.target_size
    tokens = features.flatten(2).transpose(1, 2)
    return tokens + image_encoder._build_2d_positional_encoding(
        height, width, batch_size, tokens.device, tokens.dtype
    )


def encode_shared_observation(obs_img, diffusion, disco):
    diffusion_encoder = diffusion.condition_encoder.image_encoder
    disco_encoder = disco.image_encoder
    if diffusion_encoder.use_cls_token or disco_encoder.use_cls_token:
        raise ValueError("Shared encoding requires patch-token-only image encoders.")
    if (
        diffusion_encoder.patch_size != disco_encoder.patch_size
        or diffusion_encoder.intermediate_layer_idx
        != disco_encoder.intermediate_layer_idx
    ):
        raise ValueError("Diffusion and DisCo image backbones are incompatible.")

    _, _, image_height, image_width = obs_img.shape
    patch_size = diffusion_encoder.patch_size
    padded_height = math.ceil(image_height / patch_size) * patch_size
    padded_width = math.ceil(image_width / patch_size) * patch_size
    padded = F.pad(obs_img, (0, padded_width - image_width, 0, padded_height - image_height))
    patch_tokens = diffusion_encoder.backbone.get_intermediate_layers(
        padded, [diffusion_encoder.intermediate_layer_idx], return_class_token=False
    )[0]
    grid_h, grid_w = padded_height // patch_size, padded_width // patch_size
    diffusion_tokens = project_patch_tokens(
        diffusion_encoder, patch_tokens, grid_h, grid_w
    )
    diffusion_tokens = diffusion.condition_encoder.image_mixer(
        diffusion.condition_encoder.image_norm(diffusion_tokens)
    )
    disco_tokens = project_patch_tokens(disco_encoder, patch_tokens, grid_h, grid_w)
    disco_tokens = disco.image_self_attn(disco.image_token_norm(disco_tokens))
    return diffusion_tokens, disco_tokens


def encode_map_context(condition_encoder, floorplan_img, image_tokens):
    features = condition_encoder.map_projection(
        condition_encoder.map_encoder(floorplan_img)
    )
    _, _, height, width = features.shape
    map_tokens = features.flatten(2).transpose(1, 2)
    coordinates = condition_encoder.build_map_coordinates(
        height,
        width,
        map_tokens.device,
        map_tokens.dtype,
        condition_encoder.coordinate_convention,
    )
    map_tokens = condition_encoder.map_norm(
        map_tokens + condition_encoder.map_pos_mlp(coordinates).unsqueeze(0)
    )
    attended, _ = condition_encoder.map_image_attn(
        query=map_tokens, key=image_tokens, value=image_tokens, need_weights=False
    )
    map_tokens = condition_encoder.map_image_norm(map_tokens + attended)
    map_tokens = condition_encoder.map_ffn_norm(
        map_tokens + condition_encoder.map_ffn(map_tokens)
    )
    return map_tokens, coordinates, image_tokens.mean(dim=1)


def crop_maps(floorplan, poses, crop_size_m, map_res, output_size, representation):
    return torch.stack(
        [
            crop_to_refiner_tensor(
                floorplan,
                pose,
                crop_size_meters=crop_size_m,
                map_res=map_res,
                output_size=output_size,
                representation=representation,
                oriented=True,
            ).float()
            for pose in poses.detach().cpu().numpy()
        ]
    )


def assert_shared_refiner_encoder(diffusion, refiner):
    pairs = (
        (diffusion.condition_encoder.image_encoder, refiner.refiner.image_encoder),
        (diffusion.condition_encoder.image_norm, refiner.refiner.image_norm),
        (diffusion.condition_encoder.image_mixer, refiner.refiner.image_mixer),
    )
    for diffusion_module, refiner_module in pairs:
        diffusion_state = diffusion_module.state_dict()
        refiner_state = refiner_module.state_dict()
        if diffusion_state.keys() != refiner_state.keys() or any(
            not torch.equal(diffusion_state[key].cpu(), refiner_state[key].cpu())
            for key in diffusion_state
        ):
            raise ValueError(
                "The refiner image encoder does not match the diffusion checkpoint."
            )


def load_models(args, config, device):
    config["dptv2_ckpt_path"] = args.depth_ckpt
    diffusion = PoseQueryDiffusionLocalizer.load_from_checkpoint(
        args.diffusion_ckpt, config=config, map_location="cpu"
    ).to(device).eval()

    disco_checkpoint = torch.load(
        args.disco_ckpt, map_location="cpu", weights_only=False
    )
    disco_config = dict(disco_checkpoint["hyper_parameters"])
    disco_config["dptv2_ckpt_path"] = args.depth_ckpt
    disco = DisCoLocModel.load_from_checkpoint(
        args.disco_ckpt, config=disco_config, map_location="cpu"
    ).to(device).eval()

    with open(args.refiner_config, "r", encoding="utf-8") as file:
        refiner_config = yaml.safe_load(file)
    refiner_config["baseline_checkpoint_path"] = args.diffusion_ckpt
    refiner_config["dptv2_ckpt_path"] = args.depth_ckpt
    refiner = PoseLocalRefinerLightning.load_from_checkpoint(
        args.refiner_ckpt, config=refiner_config, map_location="cpu"
    ).to(device).eval()
    assert_shared_refiner_encoder(diffusion, refiner)
    return diffusion, disco, refiner, refiner_config, disco_config


def main():
    args = parse_args()
    seed_everything(args.seed)
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    dataset_cfg = config["datasets"]
    if args.data_root:
        dataset_cfg["data_folder"] = args.data_root
    if args.split_yaml:
        dataset_cfg["data_splits"] = args.split_yaml

    dataset = DisCo_Dataset(
        data_folder=dataset_cfg["data_folder"],
        data_splits_path=dataset_cfg["data_splits"],
        split=args.split,
        floorplan_img_size=tuple(dataset_cfg["floorplan_img_size"]),
        pose_aug_params=None,
        dataset_cfg=dataset_cfg,
    )
    loader = DataLoader(
        IndexedDataset(dataset, args.max_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion, disco, refiner, refiner_config, disco_config = load_models(
        args, config, device
    )
    map_res = float(dataset_cfg["map_res"])
    representation = dataset.local_map_representation
    disco_representation = disco_config["datasets"].get(
        "local_map_representation", "gray"
    )
    if disco_representation != representation:
        raise ValueError(
            f"DisCo expects {disco_representation}, but diffusion uses "
            f"{representation}."
        )
    refiner_representation = refiner_config["datasets"].get(
        "refiner_floorplan_representation", representation
    )
    map_cache = FloorplanCache()
    metrics = {name: Metrics() for name in ("diffusion", "disco", "refined")}

    with torch.inference_mode():
        for indices, batch in tqdm(loader, desc=f"HPR-Loc {args.split}"):
            obs_img, target, _ray, floorplan_img, wh, *_ = batch
            obs_img = obs_img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            floorplan_img = floorplan_img.to(device, non_blocking=True)
            wh = wh.to(device, non_blocking=True)

            diffusion_tokens, disco_tokens = encode_shared_observation(
                obs_img, diffusion, disco
            )
            map_tokens, coordinates, image_global = encode_map_context(
                diffusion.condition_encoder, floorplan_img, diffusion_tokens
            )
            particles = diffusion.sample_from_context(
                map_tokens,
                coordinates,
                image_global,
                wh,
                num_particles=args.num_particles,
                num_steps=args.num_steps,
            )
            diffusion_pose, density = diffusion.select_pose_mode(particles)
            metrics["diffusion"].update(diffusion_pose, target, map_res)

            selected = []
            floorplans = []
            for batch_index, dataset_index in enumerate(indices.tolist()):
                mode_indices = extract_kde_modes(
                    particles[batch_index],
                    density[batch_index],
                    args.num_modes,
                    map_res,
                    args.nms_xy_m,
                    args.nms_theta_deg,
                )
                candidates = particles[batch_index, mode_indices]
                floorplan = map_cache.get(
                    dataset.data[dataset_index]["floorplan_image"],
                    representation,
                    dataset,
                )
                candidate_maps = crop_maps(
                    floorplan,
                    candidates,
                    args.disco_crop_m,
                    map_res,
                    128,
                    representation,
                ).to(device)
                candidate_images = disco_tokens[batch_index : batch_index + 1].expand(
                    len(candidates), -1, -1
                )
                disco_scores = disco.score_candidates(candidate_images, candidate_maps)
                scores = fused_candidate_scores(
                    disco_scores, density[batch_index, mode_indices], args.disco_weight
                )
                selected.append(candidates[torch.argmax(scores)])
                floorplans.append(floorplan)

            selected = torch.stack(selected)
            metrics["disco"].update(selected, target, map_res)
            local_maps = torch.stack(
                [
                    crop_maps(
                        floorplan,
                        pose.unsqueeze(0),
                        float(refiner_config["refiner_crop_size_meters"]),
                        map_res,
                        int(refiner_config["refiner_crop_output_size"]),
                        refiner_representation,
                    )[0]
                    for floorplan, pose in zip(floorplans, selected)
                ]
            ).to(device)
            output = refiner(
                obs_img=None,
                local_map=local_maps,
                candidate_pose=selected,
                wh=wh,
                image_tokens=diffusion_tokens,
            )
            refined = apply_local_delta_to_pose(
                selected,
                output["delta_xy_m"],
                output["delta_theta"],
                map_res=map_res,
                wh=wh,
            )
            metrics["refined"].update(refined, target, map_res)

    result = {
        "split": args.split,
        "seed": args.seed,
        "settings": {
            "particles": args.num_particles,
            "denoising_steps": args.num_steps,
            "kde_modes": args.num_modes,
            "nms_xy_m": args.nms_xy_m,
            "nms_theta_deg": args.nms_theta_deg,
            "disco_crop_m": args.disco_crop_m,
            "disco_weight": args.disco_weight,
            "refiner_crop_m": float(refiner_config["refiner_crop_size_meters"]),
        },
        "metrics": {name: value.summarize() for name, value in metrics.items()},
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
