import math

import torch

from eval.eval_hpr_loc import (
    extract_kde_modes,
    fused_candidate_scores,
)


def test_kde_mode_nms_keeps_distinct_translation_and_heading_modes():
    poses = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 0.0, math.pi],
        ]
    )
    density = torch.tensor([4.0, 3.0, 2.0, 1.0])
    indices = extract_kde_modes(
        poses,
        density,
        count=3,
        map_res=0.02,
        xy_radius_m=0.5,
        theta_radius_deg=30.0,
    )
    assert indices.tolist() == [0, 2, 3]


def test_fused_scores_recover_kde_and_disco_endpoints():
    disco = torch.tensor([0.0, 2.0, 1.0])
    density = torch.tensor([4.0, 2.0, 1.0])
    assert torch.argmax(fused_candidate_scores(disco, density, 0.0)).item() == 0
    assert torch.argmax(fused_candidate_scores(disco, density, 1.0)).item() == 1
