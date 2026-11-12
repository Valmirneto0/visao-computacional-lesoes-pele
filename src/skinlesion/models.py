"""Modelos (entrega final). Classificação de 7 classes com backbone do torchvision.

build_model(cfg["model"]) → nn.Module que recebe [B,3,224,224] normalizado (ImageNet) e devolve logits [B,7].
"""
import torch.nn as nn
from torchvision import models

_WEIGHTS = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V2}


def build_model(mcfg, num_classes=7):
    arch = mcfg.get("arch", "resnet18")
    pretrained = bool(mcfg.get("pretrained", True))
    net = getattr(models, arch)(weights=_WEIGHTS[arch] if pretrained else None)
    in_f = net.fc.in_features
    # cabeça nova: Dropout + Linear (512 → 7 na ResNet-18)
    net.fc = nn.Sequential(nn.Dropout(p=float(mcfg.get("dropout", 0.0))), nn.Linear(in_f, num_classes))
    return net


def count_params(net):
    total = sum(p.numel() for p in net.parameters())
    train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, train
