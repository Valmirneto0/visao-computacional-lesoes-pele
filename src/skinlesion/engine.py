"""Laço de treino/inferência e utilidades comuns (entrega final)."""
import random
import subprocess

import numpy as np
import torch
import yaml

from .data import ROOT


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device(pref=None):
    dev = torch.device(pref) if pref else torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ROCm = "cuda"
    _rocm_workarounds(dev)
    return dev


def _rocm_workarounds(dev):
    """ROCm no Windows: o MIOpen compila o kernel de BatchNorm em tempo de execução (hipRTC) e falha por falta
    de cabeçalhos C++ ('type_traits' not found). Contorno: BatchNorm usa o kernel nativo do PyTorch (MIOpen
    desligado só dentro dele); convoluções seguem no MIOpen. Se a convolução também falhar, desliga o MIOpen inteiro.
    SKIN_NO_MIOPEN=1 força o desligamento total. Não altera a matemática do modelo, só qual kernel executa."""
    import os
    import torch.nn as nn
    if dev.type != "cuda" or getattr(torch.version, "hip", None) is None:
        return
    bn = nn.modules.batchnorm._BatchNorm
    if not getattr(bn.forward, "_sem_miopen", False):
        orig = bn.forward
        def forward(self, x):
            with torch.backends.cudnn.flags(enabled=False):
                return orig(self, x)
        forward._sem_miopen = True
        bn.forward = forward
    if os.environ.get("SKIN_NO_MIOPEN") == "1":
        torch.backends.cudnn.enabled = False
        print("[ROCm] MIOpen desligado (SKIN_NO_MIOPEN=1)", flush=True); return
    try:   # teste com a rede real: forward + backward da ResNet-18 em 224 px
        from torchvision.models import resnet18
        m = resnet18(weights=None).to(dev)
        m(torch.randn(4, 3, 224, 224, device=dev)).sum().backward(); torch.cuda.synchronize()
        print("[ROCm] BatchNorm com kernel nativo; convoluções com MIOpen", flush=True)
    except RuntimeError as e:
        torch.backends.cudnn.enabled = False
        print(f"[ROCm] convolução no MIOpen também falhou ({str(e)[:60]}) → MIOpen desligado (mais lento)", flush=True)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "sem_git"


def load_cfg(path):
    return yaml.safe_load(open(ROOT / path if not str(path).startswith("/") else path, encoding="utf-8"))


def train_one_epoch(net, loader, criterion, opt, device):
    net.train(); tot, n, correct = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = net(x); loss = criterion(logits, y)
        loss.backward(); opt.step()
        tot += loss.item() * len(y); n += len(y); correct += (logits.argmax(1) == y).sum().item()
    return tot / n, correct / n


@torch.no_grad()
def predict(net, loader, device, criterion=None):
    """→ ids, y_true (np), probs (np [N,C]), loss média (se criterion)."""
    net.eval(); ids, ys, ps, tot, n = [], [], [], 0.0, 0
    for x, y, iid in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = net(x)
        if criterion is not None:
            tot += criterion(logits, y).item() * len(y)
        n += len(y)
        ps.append(torch.softmax(logits, 1).cpu().numpy()); ys.append(y.cpu().numpy()); ids += list(iid)
    return ids, np.concatenate(ys), np.concatenate(ps), (tot / n if criterion is not None else None)


def load_checkpoint(path, device):
    from .models import build_model
    ck = torch.load(path, map_location=device, weights_only=False)
    net = build_model({**ck["config"]["model"], "pretrained": False}, num_classes=len(ck["classes"]))
    net.load_state_dict(ck["state_dict"]); net.to(device).eval()
    return net, ck
