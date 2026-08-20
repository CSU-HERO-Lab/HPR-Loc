import pytest
import torch

from DisCo_model.floorplan_encoder import ResNetFloorplanEncoder


@pytest.mark.parametrize(
    ("input_mode", "input_channels"),
    (("gray", 3), ("semantic_onehot", 5)),
)
def test_paper_floorplan_inputs(input_mode, input_channels):
    encoder = ResNetFloorplanEncoder(
        feature_dim=16,
        input_mode=input_mode,
        context_blocks=1,
    )
    floorplan = torch.rand(2, input_channels, 64, 64)
    output = encoder(floorplan)
    assert output.shape == (2, 16, 8, 8)
    assert torch.isfinite(output).all()


def test_unreleased_floorplan_input_is_not_exposed():
    with pytest.raises(ValueError, match="input_mode"):
        ResNetFloorplanEncoder(input_mode="gray_ternary")
