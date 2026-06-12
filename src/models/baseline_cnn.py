"""BaselineCNN -- a custom from-scratch CNN for 96x96 traffic sign crops.

Architecture rationale (defended in the paper):

    Input: 3 x 96 x 96   (RGB, ImageNet-normalized)

    Stem  : Conv7x7/s2 -> BN -> ReLU -> MaxPool3x3/s2          [3 -> 64,   24x24]
    Block1: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
            MaxPool2x2                                         [64 -> 128, 12x12]
    Block2: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
            MaxPool2x2                                         [128 -> 256, 6x6]
    Block3: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU     [256 -> 256, 6x6]
    Head  : GlobalAvgPool -> Dropout(0.3) -> Linear(num_classes)

Design choices and justifications:
  - Conv7x7 stem with stride 2 + MaxPool collapses early redundancy fast --
    96x96 is small, aggressive early downsampling wastes signal otherwise.
  - Stacked 3x3 convs (VGG-style insight): 2 stacked 3x3 see the same field
    as one 5x5 with fewer parameters and more nonlinearity.
  - BatchNorm everywhere: stabilizes training of a randomly-initialized
    network with high class imbalance.
  - GlobalAvgPool over Flatten+Dense: ~10x fewer head parameters, less
    prone to overfitting, and accepts any spatial input size at inference.
  - Dropout 0.3 only at the head: BN already regularizes the conv stack.
  - ReLU (not GELU/SiLU): simpler, well-understood, faster on MPS.
  - ~1.4M parameters: deep enough for fine-grained signs, light enough that
    the MPS backend on an M1 Pro stays interactive (batch 128 @ 96x96).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn_relu(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class _Block(nn.Module):
    def __init__(self, in_c: int, out_c: int, downsample: bool):
        super().__init__()
        self.conv1 = conv_bn_relu(in_c, out_c)
        self.conv2 = conv_bn_relu(out_c, out_c)
        self.pool = nn.MaxPool2d(2, 2) if downsample else nn.Identity()

    def forward(self, x):
        return self.pool(self.conv2(self.conv1(x)))


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            conv_bn_relu(3, 64, k=7, s=2, p=3),        # 96 -> 48
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 48 -> 24
        )
        # Body
        self.block1 = _Block(64, 128, downsample=True)    # 24 -> 12
        self.block2 = _Block(128, 256, downsample=True)   # 12 -> 6
        self.block3 = _Block(256, 256, downsample=False)  # 6  -> 6
        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = BaselineCNN(num_classes=326)
    n = count_parameters(m)
    print(f"BaselineCNN parameters: {n:,}")
    x = torch.randn(2, 3, 96, 96)
    y = m(x)
    print(f"Output shape: {tuple(y.shape)}")
