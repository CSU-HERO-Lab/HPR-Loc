import torch

from DisCo_model.pose_query_diffusion import (
    METRIC_COORDINATE_CONVENTION,
    PoseQueryDenoiser,
    build_cell_center_coordinates,
)


def test_cached_map_key_values_preserve_denoiser_output():
    torch.manual_seed(7)
    config = {
        "diffusion_feature_dim": 32,
        "diffusion_train_steps": 100,
        "diffusion_pose_fourier_bands": 2,
        "diffusion_num_heads": 4,
        "diffusion_dropout": 0.0,
        "diffusion_denoiser_layers": 2,
        "diffusion_coordinate_convention": METRIC_COORDINATE_CONVENTION,
    }
    denoiser = PoseQueryDenoiser(config).eval()
    noisy_pose = torch.randn(2, 8, 4)
    timesteps = torch.randint(0, 100, (2, 8))
    map_tokens = torch.randn(2, 16, 32)
    map_coordinates = build_cell_center_coordinates(
        4,
        4,
        map_tokens.device,
        map_tokens.dtype,
    )
    image_global = torch.randn(2, 32)
    wh = torch.tensor([[256.0, 256.0], [320.0, 192.0]])

    uncached = denoiser(
        noisy_pose,
        timesteps,
        map_tokens,
        map_coordinates,
        image_global,
        wh,
        0.02,
    )
    cache = denoiser.build_map_kv_cache(map_tokens)
    cached = denoiser(
        noisy_pose,
        timesteps,
        map_tokens,
        map_coordinates,
        image_global,
        wh,
        0.02,
        map_kv_cache=cache,
    )

    assert len(cache) == len(denoiser.blocks)
    torch.testing.assert_close(cached, uncached, atol=0.0, rtol=0.0)
