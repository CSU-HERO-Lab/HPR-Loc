#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--monitor", required=True)
    return parser.parse_args()


def checkpoint_score(path: Path, monitor: str):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    for state in checkpoint.get("callbacks", {}).values():
        if not isinstance(state, dict) or state.get("monitor") != monitor:
            continue
        score = state.get("current_score")
        if score is not None:
            return float(score)
    return None


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    candidates = []
    for path in checkpoint_dir.glob("*.ckpt"):
        if path.name == "last.ckpt":
            continue
        score = checkpoint_score(path, args.monitor)
        if score is not None:
            candidates.append((score, path.resolve()))
    if not candidates:
        raise RuntimeError(
            f"No checkpoint monitoring {args.monitor!r} in {checkpoint_dir}."
        )
    print(max(candidates, key=lambda item: item[0])[1])


if __name__ == "__main__":
    main()
