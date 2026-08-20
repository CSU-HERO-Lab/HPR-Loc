from pathlib import Path

import pytest
import yaml
from PIL import Image

from DisCo_model.disco_dataset import DisCo_Dataset


def write_split(path: Path, train, val=None, test=None):
    path.write_text(
        yaml.safe_dump(
            {
                "train": train,
                "val": val or [],
                "test": test or [],
            }
        ),
        encoding="utf-8",
    )


def make_dataset(data_root: Path, split_path: Path):
    return DisCo_Dataset(
        data_folder=str(data_root),
        data_splits_path=str(split_path),
        split="train",
        floorplan_img_size=(32, 32),
        dataset_cfg={
            "dataset_type": "s3d",
            "map_res": 0.02,
        },
    )


def test_scene_overlap_between_splits_is_rejected(tmp_path):
    split_path = tmp_path / "split.yaml"
    write_split(
        split_path,
        train=["scene_00000"],
        val=["scene_00000"],
    )

    with pytest.raises(ValueError, match="Scene leakage"):
        make_dataset(tmp_path, split_path)


def test_missing_scene_is_reported(tmp_path):
    split_path = tmp_path / "split.yaml"
    write_split(split_path, train=["scene_00000"])

    with pytest.warns(RuntimeWarning, match="1 scene directories"):
        with pytest.raises(RuntimeError, match="No samples were loaded"):
            make_dataset(tmp_path, split_path)


def test_mismatched_scene_sample_counts_are_rejected(tmp_path):
    scene_name = "scene_00000"
    scene_dir = tmp_path / scene_name
    image_dir = scene_dir / "imgs"
    image_dir.mkdir(parents=True)

    Image.new("RGB", (32, 32), color="white").save(scene_dir / "map.png")
    Image.new("RGB", (32, 24), color="white").save(image_dir / "000.png")
    Image.new("RGB", (32, 24), color="white").save(image_dir / "001.png")
    (scene_dir / "poses_map.txt").write_text("10 10 0\n", encoding="utf-8")
    (scene_dir / "depth40.txt").write_text(
        " ".join(["1"] * 40) + "\n",
        encoding="utf-8",
    )

    split_path = tmp_path / "split.yaml"
    write_split(split_path, train=[scene_name])

    with pytest.raises(ValueError, match="mismatched sample counts"):
        make_dataset(tmp_path, split_path)
