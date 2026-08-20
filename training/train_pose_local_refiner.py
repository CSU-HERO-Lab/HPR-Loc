import argparse
import os
import sys
from datetime import datetime

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DisCo_model.pose_local_refiner import PoseLocalRefinerLightning, PoseRefinerDataset
from training.checkpoint_utils import update_best_checkpoint_link


def build_dataset(config: dict, split: str, deterministic: bool):
    dataset_cfg = config["datasets"]
    return PoseRefinerDataset(
        dataset_cfg=dataset_cfg,
        split=split,
        floorplan_img_size=tuple(dataset_cfg["floorplan_img_size"]),
        crop_size_meters=float(config.get("refiner_crop_size_meters", 5.0)),
        crop_output_size=int(config.get("refiner_crop_output_size", 256)),
        max_delta_m=float(config.get("refiner_max_delta_m", 1.5)),
        max_delta_theta_deg=float(config.get("refiner_max_delta_theta_deg", 45.0)),
        score_sigma_m=float(config.get("refiner_score_sigma_m", 0.5)),
        score_sigma_deg=float(config.get("refiner_score_sigma_deg", 20.0)),
        deterministic=deterministic,
        seed=int(config.get("refiner_val_seed", 42)),
    )


def build_validation_subset(dataset, dataset_cfg):
    fraction = float(dataset_cfg.get("val_subset_fraction", 1.0))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("datasets.val_subset_fraction must be in (0, 1].")
    if fraction >= 1.0:
        return dataset

    subset_size = max(1, int(round(len(dataset) * fraction)))
    generator = torch.Generator().manual_seed(
        int(dataset_cfg.get("val_subset_seed", 42))
    )
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    indices = indices.sort().values.tolist()
    print(
        f"Using fixed validation subset: {subset_size}/{len(dataset)} "
        f"samples ({fraction:.1%})."
    )
    return Subset(dataset, indices)


def main(config, ckpt_path=None):
    pl.seed_everything(int(config.get("seed", 42)), workers=True)
    num_workers = int(config.get("num_workers", 4))
    val_split = config["datasets"].get("val_split", "val")
    if val_split == "test":
        raise ValueError(
            "Training may not use the test split for model selection. "
            "Set datasets.val_split to 'val'."
        )
    train_dataset = build_dataset(config, split="train", deterministic=False)
    val_dataset = build_dataset(
        config,
        split=val_split,
        deterministic=True,
    )
    val_dataset = build_validation_subset(val_dataset, config["datasets"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("val_batch_size", config["batch_size"])),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = PoseLocalRefinerLightning(config)
    run_dir = os.path.join("logs", "pose_diffusion_runs", config["run_name"])
    os.makedirs(run_dir, exist_ok=True)

    logger = True
    if config.get("use_wandb", False):
        logger = WandbLogger(
            project=config.get("project_name", "hpr-loc"),
            name=config["run_name"],
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = config.get("checkpoint_dir", os.path.join(run_dir, "checkpoints"))
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_refined_0.5m_recall:.3f}-"
        "{val_refined_0.1m_recall:.3f}_"
        + timestamp,
        save_top_k=3,
        save_last=True,
        monitor="val_refined_0.5m_recall",
        mode="max",
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.get("gpu_ids", 1),
        max_epochs=int(config.get("epochs", 30)),
        callbacks=[checkpoint],
        logger=logger,
        default_root_dir=run_dir,
        log_every_n_steps=int(config.get("log_every_n_steps", 10)),
        precision=config.get("precision", "32-true"),
        gradient_clip_val=float(config.get("gradient_clip_val", 1.0)),
        limit_train_batches=config.get("limit_train_batches", None),
        limit_val_batches=config.get("limit_val_batches", None),
        num_sanity_val_steps=int(config.get("num_sanity_val_steps", 1)),
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    update_best_checkpoint_link(checkpoint.best_model_path, checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--run_name")
    parser.add_argument("--baseline_checkpoint_path")
    parser.add_argument("--ckpt_path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.run_name:
        config["run_name"] = args.run_name
    if args.baseline_checkpoint_path:
        config["baseline_checkpoint_path"] = args.baseline_checkpoint_path
    main(config, ckpt_path=args.ckpt_path)
