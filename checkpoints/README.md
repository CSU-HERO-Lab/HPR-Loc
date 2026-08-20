# Checkpoints

Model weights are excluded from Git. Place the pretrained image backbone at:

```text
checkpoints/depth_anything_v2_vits.pth
```

The default training configurations write selected HPR-Loc checkpoints to:

```text
checkpoints/hpr_<dataset>_<representation>/diffusion/best.ckpt
checkpoints/hpr_<dataset>_<representation>/disco/best.ckpt
checkpoints/hpr_<dataset>_<representation>/refiner/best.ckpt
```

Here `<dataset>` is `s3d` or `zind`, and `<representation>` is `gray` or
`semantic`.
