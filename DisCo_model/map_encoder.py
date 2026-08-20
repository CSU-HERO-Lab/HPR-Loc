import torch
import torch.nn as nn
import torchvision.models as models


class MapEncoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=128, use_pretrained=True):
        super().__init__()

        # Load the ResNet-18 backbone.
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Adapt the input stem while preserving pretrained RGB statistics.
        if input_channels != 3:
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                input_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False,
            )
            if use_pretrained:
                with torch.no_grad():
                    channel_mean = original_conv1.weight.data.mean(
                        dim=1, keepdim=True
                    )
                    self.backbone.conv1.weight.copy_(
                        channel_mean.expand(-1, input_channels, -1, -1)
                    )

        # Preserve the spatial feature map and project it to the model dimension.
        self.proj_conv = nn.Conv2d(512, feature_dim, kernel_size=1)

    def forward(self, x):
        # Forward explicitly to retain the layer-4 spatial feature map.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.proj_conv(x)
        return x
