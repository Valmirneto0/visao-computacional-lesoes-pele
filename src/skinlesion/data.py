"""Datasets e transformações (M2).

Contrato:
- Tudo lê de data/processed/ (gerado por scripts/preprocess.py); o split vem de configs/data.yaml → split_file
  (data/splits/split_v2.csv) e sobrescreve a coluna split do manifest.
- Transformações vêm de configs/transforms.yaml; nenhuma augmentation é definida em código.
- ClassificationDataset → (imagem[3,224,224] float, rótulo int, id)
- SegmentationDataset   → (imagem[3,256,256] float, máscara[1,256,256] float 0/1, id)
"""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(rel):
    return yaml.safe_load(open(ROOT / rel))


# ---------------------------------------------------------------- transforms
_BORDER = {"reflect": cv2.BORDER_REFLECT_101, "constant": cv2.BORDER_CONSTANT}


def _build_one(spec):
    spec = dict(spec); name = spec.pop("name")
    if "border_mode" in spec:
        spec["border_mode"] = _BORDER[spec["border_mode"]]
    if name == "RandomResizedCrop":
        spec["size"] = tuple(spec["size"])
    if name == "ShiftScaleRotate":  # API nova do albumentations 2.x usa Affine; mantém nome legível no YAML
        return A.Affine(translate_percent=spec["shift_limit"], scale=(1 - spec["scale_limit"], 1 + spec["scale_limit"]),
                        rotate=(-spec["rotate_limit"], spec["rotate_limit"]), border_mode=spec["border_mode"], p=spec["p"])
    return getattr(A, name)(**spec)


def build_transforms(task, split, cfg=None):
    """task: 'classification' | 'segmentation'; split: 'train' | 'val' | 'test' | 'test_official'."""
    cfg = cfg or load_yaml("configs/transforms.yaml")
    phase = "train" if split == "train" else "eval"
    steps = [_build_one(s) for s in cfg[task][phase]]
    steps += [A.Normalize(mean=cfg["normalize"]["mean"], std=cfg["normalize"]["std"]), ToTensorV2()]
    return A.Compose(steps)


# ---------------------------------------------------------------- datasets
def mask_path(img_rel):
    """data/processed/seg/images/X.jpg → data/processed/seg/masks/X.png (independente do separador)."""
    p = Path(str(img_rel).replace("\\", "/"))
    return p.parent.parent / "masks" / (p.stem + ".png")


def load_manifest(task, split, data_cfg=None, pre_cfg=None):
    pre_cfg = pre_cfg or load_yaml("configs/preprocess.yaml")
    m = pd.read_csv(ROOT / pre_cfg["output"]["manifest"])
    m["path"] = m.path.str.replace("\\", "/", regex=False)
    data_cfg = data_cfg or load_yaml("configs/data.yaml")
    if "split_file" in data_cfg:  # fonte da verdade do split = arquivo de split versionado
        sp = pd.read_csv(ROOT / data_cfg["split_file"]); smap = dict(zip(sp.image, sp.split))
        m["split"] = [("test_official" if s == "test_official" else smap.get(i, "NA")) for i, s in zip(m.id, m.source)]
    m = m[(m.task == task) & (m.split == split) & (m.status.isin(["ok", "skip"]))]
    if len(m) == 0:
        raise RuntimeError(f"manifest vazio para task={task} split={split} — rode scripts/preprocess.py")
    return m.reset_index(drop=True)


class ClassificationDataset(Dataset):
    def __init__(self, split, transforms=None, data_cfg=None):
        data_cfg = data_cfg or load_yaml("configs/data.yaml")
        self.classes = data_cfg["classes"]; self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.m = load_manifest("cls", split)
        # rótulos: split interno usa o GT de treino; test_official usa o GT do teste do desafio
        gt_path = data_cfg["task3_test_labels"] if split == "test_official" else data_cfg["task3_labels"]
        gt = pd.read_csv(ROOT / gt_path); gt.columns = [c.strip() for c in gt.columns]
        gt["dx"] = gt[self.classes].idxmax(axis=1)
        self.m = self.m.merge(gt[["image", "dx"]], left_on="id", right_on="image", how="left")
        assert self.m.dx.notna().all(), "imagem sem rótulo no manifest"
        self.m["y"] = self.m.dx.map(self.class_to_idx)
        self.t = transforms if transforms is not None else build_transforms("classification", split)

    def __len__(self): return len(self.m)

    def __getitem__(self, i):
        r = self.m.iloc[i]
        img = cv2.cvtColor(cv2.imread(str(ROOT / r.path)), cv2.COLOR_BGR2RGB)
        return self.t(image=img)["image"], int(r.y), r.id

    def class_counts(self):
        return self.m.dx.value_counts().reindex(self.classes).fillna(0).astype(int)


class SegmentationDataset(Dataset):
    def __init__(self, split, transforms=None):
        self.m = load_manifest("seg", split)
        self.t = transforms if transforms is not None else build_transforms("segmentation", split)

    def __len__(self): return len(self.m)

    def __getitem__(self, i):
        r = self.m.iloc[i]
        img = cv2.cvtColor(cv2.imread(str(ROOT / r.path)), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(ROOT / mask_path(r.path)), cv2.IMREAD_GRAYSCALE)
        assert msk is not None, f"máscara não encontrada para {r.id}"
        out = self.t(image=img, mask=msk)               # mesma transformação geométrica nos dois
        mask = (out["mask"] > 127).float().unsqueeze(0)  # [1,H,W] em {0,1}
        return out["image"], mask, r.id


def denormalize(t, cfg=None):
    """Tensor [3,H,W] normalizado → uint8 HxWx3 (para visualizar augmentations)."""
    cfg = cfg or load_yaml("configs/transforms.yaml")
    mean = torch.tensor(cfg["normalize"]["mean"]).view(3, 1, 1); std = torch.tensor(cfg["normalize"]["std"]).view(3, 1, 1)
    x = (t * std + mean).clamp(0, 1) * 255
    return x.permute(1, 2, 0).numpy().astype(np.uint8)
