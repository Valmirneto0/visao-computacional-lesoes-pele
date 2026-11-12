"""Funções de perda (entrega final)."""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ROOT


def load_class_weights(path, classes):
    w = json.load(open(ROOT / path))["pesos"]
    return torch.tensor([w[c] for c in classes], dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal loss multiclasse (Lin et al., 2017): FL = -alpha_y * (1 - p_y)^gamma * log(p_y).
    gamma = 0 com os mesmos alpha reduz exatamente à CE ponderada do baseline — por isso a comparação é controlada.
    Normalização: média ponderada (soma / soma dos alpha_y), igual à CrossEntropyLoss(weight=...) do PyTorch."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits, y):
        logp = F.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        p = logp.exp()
        a = self.alpha[y] if self.alpha is not None else torch.ones_like(p)
        loss = -a * (1 - p) ** self.gamma * logp
        return loss.sum() / a.sum()


def build_loss(lcfg, classes):
    w = load_class_weights(lcfg["class_weights"], classes) if lcfg.get("class_weights") else None
    if lcfg["name"] == "ce":
        return nn.CrossEntropyLoss(weight=w)
    if lcfg["name"] == "focal":
        return FocalLoss(alpha=w, gamma=float(lcfg.get("gamma", 2.0)))
    raise ValueError(lcfg["name"])
