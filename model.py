"""
감정 분류 모델

지원 backbone: efficientnet_b0 | densenet121 | densenet169 | resnet18 | resnet50
in_channels=4 시 엣지 채널 지원 (첫 번째 conv 가중치 확장)
"""
import torch
import torch.nn as nn
from torchvision import models


def _adapt_first_conv(model, backbone: str, in_channels: int):
    """in_channels=4일 때 첫 번째 conv를 3→4ch로 확장. pretrained 가중치 보존."""
    if in_channels == 3:
        return model

    if backbone == 'efficientnet_b0':
        old = model.features[0][0]
        new = nn.Conv2d(4, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3]  = old.weight.mean(dim=1)
        model.features[0][0] = new

    elif backbone.startswith('densenet'):
        old = model.features.conv0
        new = nn.Conv2d(4, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3]  = old.weight.mean(dim=1)
        model.features.conv0 = new

    elif backbone.startswith('resnet'):
        old = model.conv1
        new = nn.Conv2d(4, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3]  = old.weight.mean(dim=1)
        model.conv1 = new

    return model


def build_model(num_classes: int, backbone: str = 'efficientnet_b0',
                pretrained: bool = True, in_channels: int = 3):
    weights = 'DEFAULT' if pretrained else None

    if backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, num_classes))

    elif backbone == 'densenet121':
        model = models.densenet121(weights=weights)
        in_feat = model.classifier.in_features      # 1024
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, num_classes))

    elif backbone == 'densenet169':
        model = models.densenet169(weights=weights)
        in_feat = model.classifier.in_features      # 1664
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, num_classes))

    elif backbone == 'resnet18':
        model = models.resnet18(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, num_classes))

    elif backbone == 'resnet50':
        model = models.resnet50(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, num_classes))

    else:
        raise ValueError(f'지원하지 않는 backbone: {backbone}')

    if in_channels != 3:
        model = _adapt_first_conv(model, backbone, in_channels)

    return model


class EmotionClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone: str = 'efficientnet_b0',
                 pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.net = build_model(num_classes, backbone, pretrained, in_channels)

    def forward(self, x):
        return self.net(x)
