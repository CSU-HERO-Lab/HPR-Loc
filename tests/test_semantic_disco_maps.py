import numpy as np
import torch

from DisCo_model.disco_dataset import DisCo_Dataset
from DisCo_model.map_encoder import MapEncoder


def test_semantic_local_crop_is_five_channel_onehot():
    dataset = DisCo_Dataset.__new__(DisCo_Dataset)
    dataset.map_res = 0.1
    dataset.local_map_representation = "semantic_onehot"
    labels = np.zeros((80, 80), dtype=np.uint8)
    labels[:, 40:] = 1
    labels[30:50, 30:50] = 3

    crop = dataset.crop_local_map_tensor(
        labels,
        pose=np.array([40.0, 40.0, 0.0], dtype=np.float32),
        crop_size_meters=5.0,
    )

    assert crop.shape == (5, 128, 128)
    assert torch.isfinite(crop).all()
    assert torch.equal(crop.sum(dim=0), torch.ones(128, 128))
    assert set(torch.unique(crop).tolist()) <= {0.0, 1.0}


def test_map_encoder_accepts_five_channel_input():
    encoder = MapEncoder(input_channels=5, feature_dim=16, use_pretrained=False)
    output = encoder(torch.rand(2, 5, 128, 128))
    assert output.shape == (2, 16, 4, 4)
    assert torch.isfinite(output).all()
