from pathlib import Path

import yaml


PROTOCOLS = ("s3d_gray", "s3d_semantic", "zind_gray", "zind_semantic")


def load_config(protocol, stage):
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "configs" / protocol / f"{stage}.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_release_protocols_use_paper_defaults():
    for protocol in PROTOCOLS:
        diffusion = load_config(protocol, "diffusion")
        disco = load_config(protocol, "disco")
        refiner = load_config(protocol, "refiner")

        assert diffusion["seed"] == disco["seed"] == refiner["seed"] == 42
        assert diffusion["diffusion_val_particles"] == 64
        assert diffusion["diffusion_sample_steps"] == 10
        assert disco["datasets"]["local_map_crop_size_meters"] == 7.0
        assert refiner["refiner_crop_size_meters"] == 11.0
        assert refiner["datasets"]["refiner_oriented_crop"] is True


def test_semantic_models_train_longer_than_grayscale_models():
    for dataset in ("s3d", "zind"):
        assert load_config(f"{dataset}_gray", "diffusion")["epochs"] == 30
        assert load_config(f"{dataset}_semantic", "diffusion")["epochs"] == 60
