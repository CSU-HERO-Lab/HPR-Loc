import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class ResNetFloorplanEncoder(nn.Module):
    """Floorplan encoder for the map representations used in paper results."""

    INPUT_CHANNELS = {
        "gray": 1,
        "semantic_onehot": 5,
    }

    def __init__(
        self,
        feature_dim: int = 64,
        input_mode: str = "gray",
        context_blocks: int = 1,
        pretrained: bool = False,
    ):
        super().__init__()
        if input_mode not in self.INPUT_CHANNELS:
            raise ValueError(
                "input_mode must be one of "
                f"{sorted(self.INPUT_CHANNELS)}; got {input_mode!r}."
            )

        self.input_mode = input_mode
        input_channels = self.INPUT_CHANNELS[input_mode]
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        if input_channels != 3:
            original_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    mean_weight = original_conv.weight.mean(dim=1, keepdim=True)
                    backbone.conv1.weight.copy_(
                        mean_weight.repeat(1, input_channels, 1, 1)
                        / input_channels
                    )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.proj = nn.Conv2d(128, feature_dim, kernel_size=1, bias=False)
        self.proj_norm = nn.GroupNorm(min(8, feature_dim), feature_dim)
        self.context = nn.Sequential(
            *[ResidualConvBlock(feature_dim) for _ in range(max(0, context_blocks))]
        )

    def forward(self, floorplan_img: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "gray" and floorplan_img.shape[1] == 3:
            floorplan_img = floorplan_img.mean(dim=1, keepdim=True)
        floorplan_img = floorplan_img.float()

        expected_channels = self.INPUT_CHANNELS[self.input_mode]
        if floorplan_img.shape[1] != expected_channels:
            raise ValueError(
                f"{self.input_mode} expects {expected_channels} channels after "
                f"preprocessing, got {floorplan_img.shape[1]}."
            )

        features = self.relu(self.bn1(self.conv1(floorplan_img)))
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = F.gelu(self.proj_norm(self.proj(features)))
        return self.context(features)
