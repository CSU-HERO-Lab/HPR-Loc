from pathlib import Path

import torch
import torch.nn as nn
import yaml

from DisCo_model.pose_query_diffusion import (
    PoseQueryDiffusionLocalizer,
    cosine_beta_schedule,
)


class DummyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, noisy_pose, *_args):
        return self.projection(noisy_pose)


def build_lightweight_localizer():
    model = PoseQueryDiffusionLocalizer.__new__(PoseQueryDiffusionLocalizer)
    nn.Module.__init__(model)
    model.map_res = 0.02
    model.num_train_steps = 16
    model.train_particles = 2
    model.theta_loss_weight = 1.0
    model.denoiser = DummyDenoiser()

    betas = cosine_beta_schedule(model.num_train_steps)
    alpha_cumprod = torch.cumprod(1.0 - betas, dim=0)
    model.register_buffer("sqrt_alpha_cumprod", alpha_cumprod.sqrt())
    model.register_buffer(
        "sqrt_one_minus_alpha_cumprod",
        (1.0 - alpha_cumprod).sqrt(),
    )
    return model


def test_diffusion_objective_is_noise_only():
    model = build_lightweight_localizer()
    pose = torch.tensor([[64.0, 96.0, 0.5], [128.0, 48.0, 2.0]])
    wh = torch.full((2, 2), 256.0)
    map_tokens = torch.randn(2, 4, 8)
    map_coordinates = torch.randn(4, 2)
    image_global = torch.randn(2, 8)

    loss, parts = model.diffusion_loss(
        pose,
        wh,
        map_tokens,
        map_coordinates,
        image_global,
    )

    assert set(parts) == {"xy_noise_loss", "theta_noise_loss"}
    assert torch.allclose(
        loss,
        parts["xy_noise_loss"] + parts["theta_noise_loss"],
    )
    loss.backward()
    assert model.denoiser.projection.weight.grad is not None
    assert torch.isfinite(model.denoiser.projection.weight.grad).all()


def test_main_configs_do_not_expose_reconstructed_pose_loss():
    root = Path(__file__).resolve().parents[1]
    expected_map_modes = {
        "configs/s3d_gray/diffusion.yaml": "gray",
        "configs/zind_gray/diffusion.yaml": "gray",
        "configs/s3d_semantic/diffusion.yaml": "semantic_onehot",
        "configs/zind_semantic/diffusion.yaml": "semantic_onehot",
    }
    for filename, expected_map_mode in expected_map_modes.items():
        config = yaml.safe_load((root / filename).read_text(encoding="utf-8"))
        assert config["diffusion_map_input_mode"] == expected_map_mode
        assert all("clean" not in key for key in config)
