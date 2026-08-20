import torch

from DisCo_model.pose_local_refiner import DensePoseLocalRefiner


def test_dense_coordinates_follow_oriented_crop_axes():
    coordinates = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    local_m = DensePoseLocalRefiner.coordinates_to_local_m(coordinates, 5.0)
    expected = torch.tensor(
        [
            [2.5, 0.0],
            [0.0, 2.5],
            [-2.5, 0.0],
            [0.0, -2.5],
        ]
    )
    torch.testing.assert_close(local_m, expected)


def test_dense_grid_coordinates_are_cell_centered():
    coordinates = DensePoseLocalRefiner.build_dense_coordinates(
        2, 2, torch.device("cpu"), torch.float32
    )
    expected = torch.tensor(
        [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
    )
    torch.testing.assert_close(coordinates, expected)


def test_dense_soft_argmax_recovers_peaked_cell():
    coordinates = DensePoseLocalRefiner.build_dense_coordinates(
        5, 5, torch.device("cpu"), torch.float32
    )
    local_m = DensePoseLocalRefiner.coordinates_to_local_m(coordinates, 5.0)
    target_index = 7
    logits = torch.full((1, 25), -20.0)
    logits[0, target_index] = 20.0
    prediction = torch.einsum("bn,nd->bd", logits.softmax(-1), local_m)
    torch.testing.assert_close(prediction[0], local_m[target_index], atol=1e-5, rtol=0.0)
