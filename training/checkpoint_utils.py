import os
from pathlib import Path


def update_best_checkpoint_link(best_model_path: str, checkpoint_dir: str) -> str:
    if not best_model_path:
        raise RuntimeError("Training finished without a best checkpoint.")
    checkpoint_dir = Path(checkpoint_dir).resolve()
    best_model_path = Path(best_model_path).resolve()
    stable_path = checkpoint_dir / "best.ckpt"
    temporary_path = checkpoint_dir / ".best.ckpt.tmp"
    if temporary_path.exists() or temporary_path.is_symlink():
        temporary_path.unlink()
    temporary_path.symlink_to(os.path.relpath(best_model_path, checkpoint_dir))
    temporary_path.replace(stable_path)
    print(f"BEST_CHECKPOINT={best_model_path}")
    print(f"BEST_CHECKPOINT_LINK={stable_path}")
    return str(stable_path)
