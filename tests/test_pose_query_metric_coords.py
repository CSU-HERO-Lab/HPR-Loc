import math
import unittest

import torch

from DisCo_model.pose_query_diffusion import (
    ImageConditionedMapEncoder,
    LEGACY_COORDINATE_CONVENTION,
    METRIC_COORDINATE_CONVENTION,
    PoseMapCrossAttention,
    build_cell_center_coordinates,
    map_xy_to_normalized,
    normalized_to_map_xy,
)


class MetricCoordinateTest(unittest.TestCase):
    def test_pose_round_trip_and_boundaries(self):
        wh = torch.tensor([[640.0, 480.0], [1200.0, 800.0]])
        xy = torch.tensor([[0.0, 480.0], [1200.0, 0.0]])
        normalized = map_xy_to_normalized(xy, wh)
        expected = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])
        torch.testing.assert_close(normalized, expected)
        torch.testing.assert_close(normalized_to_map_xy(normalized, wh), xy)

        interior = torch.tensor(
            [
                [[0.0, 0.0], [319.25, 127.75]],
                [[15.5, 20.25], [1199.9, 799.9]],
            ]
        )
        round_trip = normalized_to_map_xy(
            map_xy_to_normalized(interior, wh), wh
        )
        torch.testing.assert_close(round_trip, interior, atol=1e-4, rtol=0.0)

    def test_feature_tokens_are_cell_centers(self):
        coordinates = build_cell_center_coordinates(
            height=2,
            width=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [
                [-0.75, -0.5],
                [-0.25, -0.5],
                [0.25, -0.5],
                [0.75, -0.5],
                [-0.75, 0.5],
                [-0.25, 0.5],
                [0.25, 0.5],
                [0.75, 0.5],
            ]
        )
        torch.testing.assert_close(coordinates, expected)
        torch.testing.assert_close(
            ImageConditionedMapEncoder.build_map_coordinates(
                2,
                4,
                torch.device("cpu"),
                torch.float32,
                METRIC_COORDINATE_CONVENTION,
            ),
            expected,
        )

    def test_legacy_feature_coordinates_keep_endpoint_grid(self):
        coordinates = ImageConditionedMapEncoder.build_map_coordinates(
            2,
            4,
            torch.device("cpu"),
            torch.float32,
            LEGACY_COORDINATE_CONVENTION,
        )
        expected = torch.tensor(
            [
                [-1.0, -1.0],
                [-1.0 / 3.0, -1.0],
                [1.0 / 3.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0 / 3.0, 1.0],
                [1.0 / 3.0, 1.0],
                [1.0, 1.0],
            ]
        )
        torch.testing.assert_close(coordinates, expected)

    def test_metric_features_are_map_size_invariant(self):
        # Both particles are one meter left of the same map token. Their maps
        # have different widths, so the normalized offsets differ by 2x.
        map_coordinates = torch.tensor([[0.0, 0.0]])
        noisy_pose = torch.tensor(
            [
                [[-0.2, 0.0, 0.0, 1.0]],
                [[-0.4, 0.0, 0.0, 1.0]],
            ]
        )
        wh = torch.tensor([[1000.0, 600.0], [500.0, 900.0]])
        features = PoseMapCrossAttention.build_metric_relative_features(
            noisy_pose=noisy_pose,
            map_coordinates=map_coordinates,
            wh=wh,
            map_res=0.01,
            max_distance_m=20.0,
            scale_m=5.0,
        )
        torch.testing.assert_close(features[0], features[1])
        self.assertAlmostEqual(features[0, 0, 0, 0].item(), 0.2, places=6)
        self.assertAlmostEqual(features[0, 0, 0, 1].item(), 0.0, places=6)
        self.assertAlmostEqual(features[0, 0, 0, 2].item(), 0.2, places=6)

    def test_heading_rotates_metric_offset(self):
        map_coordinates = torch.tensor([[0.2, 0.0]])
        noisy_pose = torch.tensor(
            [[[0.0, 0.0, math.sin(math.pi / 2), math.cos(math.pi / 2)]]]
        )
        features = PoseMapCrossAttention.build_metric_relative_features(
            noisy_pose,
            map_coordinates,
            torch.tensor([[1000.0, 1000.0]]),
            0.01,
            20.0,
            5.0,
        )
        self.assertAlmostEqual(features[0, 0, 0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(features[0, 0, 0, 1].item(), -0.2, places=6)


if __name__ == "__main__":
    unittest.main()
