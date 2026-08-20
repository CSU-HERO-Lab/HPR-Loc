# HPR-Loc

Official implementation of **HPR-Loc**, a hierarchical visual floorplan
localization framework. HPR-Loc combines three complementary components:

1. a full-map pose diffusion model for global multi-modal localization;
2. a DisCo matcher that reranks distinct KDE pose modes using local
   image-floorplan correspondence;
3. an oriented dense local refiner for sub-meter pose correction.

The release supports grayscale and semantic one-hot floorplans on
Structured3D and ZInD.

## Installation

HPR-Loc is tested with Python 3.8 and PyTorch 2.4.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Depth Anything V2 ViT-S checkpoint to:

```text
checkpoints/depth_anything_v2_vits.pth
```

The checkpoint is available from the
[Depth Anything V2 repository](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth).

Dataset preparation and directory layouts are described in
[`docs/DATASETS.md`](docs/DATASETS.md).

## Configurations

| Protocol | Diffusion | DisCo | Refiner |
| --- | --- | --- | --- |
| S3D grayscale | `configs/s3d_gray/diffusion.yaml` | `configs/s3d_gray/disco.yaml` | `configs/s3d_gray/refiner.yaml` |
| S3D semantic | `configs/s3d_semantic/diffusion.yaml` | `configs/s3d_semantic/disco.yaml` | `configs/s3d_semantic/refiner.yaml` |
| ZInD grayscale | `configs/zind_gray/diffusion.yaml` | `configs/zind_gray/disco.yaml` | `configs/zind_gray/refiner.yaml` |
| ZInD semantic | `configs/zind_semantic/diffusion.yaml` | `configs/zind_semantic/disco.yaml` | `configs/zind_semantic/refiner.yaml` |

The released defaults use seed 42, 64 diffusion particles, 10 denoising
steps, five KDE modes, a 7 m DisCo crop, DisCo fusion weight 0.9, and an
11 m oriented refiner crop.

## Training

Choose one protocol directory and train its three components in order:

```bash
python training/train_pose_query_diffusion.py \
  --config configs/s3d_gray/diffusion.yaml

python training/train_disco_model.py \
  --config configs/s3d_gray/disco.yaml

python training/train_pose_local_refiner.py \
  --config configs/s3d_gray/refiner.yaml
```

Set `baseline_checkpoint_path` in the refiner config to the selected diffusion
checkpoint. Each trainer writes `best.ckpt` into its configured checkpoint
directory.

## Evaluation

Evaluate the complete pipeline on the test split:

```bash
python eval/eval_hpr_loc.py \
  --config configs/s3d_gray/diffusion.yaml \
  --refiner-config configs/s3d_gray/refiner.yaml \
  --diffusion-ckpt checkpoints/hpr_s3d_gray/diffusion/best.ckpt \
  --disco-ckpt checkpoints/hpr_s3d_gray/disco/best.ckpt \
  --refiner-ckpt checkpoints/hpr_s3d_gray/refiner/best.ckpt \
  --split test
```

The evaluator reports results after diffusion mode selection, after DisCo
reranking, and after local refinement.

## Tests

```bash
pytest -q
ruff check .
```

## License

HPR-Loc is released under the MIT License. Vendored Depth Anything V2 and
DINOv2 components retain their Apache-2.0 terms; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
