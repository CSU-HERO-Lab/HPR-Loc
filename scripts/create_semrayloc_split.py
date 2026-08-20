import argparse
import re
from pathlib import Path

import yaml
from PIL import Image


def has_required_files(scene_dir: Path) -> bool:
    if not (
        scene_dir.is_dir()
        and (scene_dir / "floorplan_semantic.png").exists()
        and (scene_dir / "poses.txt").exists()
        and (scene_dir / "depth.txt").exists()
        and (scene_dir / "rgb").is_dir()
    ):
        return False
    try:
        with Image.open(scene_dir / "floorplan_semantic.png") as img:
            img.verify()
    except Exception:
        return False
    return True


def scene_sort_key(path: Path):
    suffix = path.name.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return (path.name[: -len(suffix)], int(suffix))
    return (path.name, -1)


def scene_id_from_name(name: str) -> int:
    digits = "".join(re.findall(r"\d", str(name)))
    if not digits:
        raise ValueError(f"Cannot parse scene id from {name!r}")
    return int(digits[-5:])


def load_reference_scene_ids(reference_split: Path):
    raw_split = yaml.safe_load(reference_split.read_text())
    split = {}
    for split_name in ("train", "val", "test"):
        split[split_name] = [
            scene_id_from_name(scene_name)
            for scene_name in raw_split[split_name]
        ]
    return split


def main():
    parser = argparse.ArgumentParser(
        description="Create a SemRayLoc split aligned to the existing S3D baseline split."
    )
    parser.add_argument("--data-folder", required=True, type=Path)
    parser.add_argument(
        "--reference-split",
        default=Path("datasets_s3d/Structured3D/split.yaml"),
        type=Path,
        help="S3D baseline split to mirror by scene id.",
    )
    parser.add_argument(
        "--output",
        default=Path("datasets_semrayloc/split.yaml"),
        type=Path,
    )
    args = parser.parse_args()

    valid_scenes = {
        p.name: p
        for p in args.data_folder.iterdir()
        if p.is_dir() and has_required_files(p)
    }
    if not valid_scenes:
        raise RuntimeError(f"No valid SemRayLoc scenes found in {args.data_folder}")

    reference_split = load_reference_scene_ids(args.reference_split)

    split = {}
    missing = {}
    for split_name, scene_ids in reference_split.items():
        scene_names = []
        missing_names = []
        for scene_id in scene_ids:
            unpadded = f"scene_{scene_id}"
            padded = f"scene_{scene_id:05d}"
            scene_name = unpadded if unpadded in valid_scenes else padded
            if scene_name in valid_scenes:
                scene_names.append(scene_name)
            else:
                missing_names.append(unpadded)
        split[split_name] = sorted(scene_names, key=lambda name: scene_sort_key(valid_scenes[name]))
        missing[split_name] = missing_names

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(split, f, sort_keys=False)

    print(
        f"Wrote {args.output}: "
        f"train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}"
    )
    for split_name in ("train", "val", "test"):
        if missing[split_name]:
            examples = ", ".join(missing[split_name][:10])
            suffix = "..." if len(missing[split_name]) > 10 else ""
            print(f"Missing {split_name}: {len(missing[split_name])} ({examples}{suffix})")


if __name__ == "__main__":
    main()
