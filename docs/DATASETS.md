# Datasets

HPR-Loc uses a shared scene layout across training stages. Dataset paths are
set in the selected YAML configuration and may be regular directories or
symbolic links.

## Structured3D Grayscale

```text
datasets_s3d/Structured3D/
  split.yaml
  scene_00000/
    imgs/*.png
    map.png
    poses_map.txt
    depth40.txt
```

The grayscale protocol uses 0.02 m per map pixel.

## Structured3D Semantic

The semantic protocol uses the SemRayLoc-compatible conversion and the same
train/validation/test scene split:

```text
datasets_semrayloc/
  split.yaml
  processed/scene_0/
    rgb/*.png
    floorplan_semantic.png
    poses.txt
    depth.txt
```

Semantic colors are converted to five hard one-hot channels by the dataset
loader. The map resolution is 0.01 m per pixel.

## ZInD

Prepare grayscale and semantic ZInD inputs with:

```bash
python scripts/prepare_zind.py \
  --raw-root datasets_zind/raw_data \
  --output-root datasets_zind/disco_floc \
  --semrayloc-processed-root datasets_zind/processed \
  --skip-existing
```

The generated grayscale data are stored under `datasets_zind/disco_floc`.
Semantic configurations read aligned floorplans from
`datasets_zind/processed`. Both protocols use the same home-disjoint split and
0.01 m per map pixel.

Training scripts reject `datasets.val_split: test`. Use validation for model
selection and pass `--split test` only to the evaluator for final reporting.
