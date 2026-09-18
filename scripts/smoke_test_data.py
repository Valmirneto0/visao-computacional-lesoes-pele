"""Smoke test do pipeline de dados (M2): carrega um batch de cada dataset/split e confere formas,
faixas de valores e se vai para a GPU. Roda em ~10 s. Uso: python scripts/smoke_test_data.py
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch.utils.data import DataLoader

from skinlesion.data import ClassificationDataset, SegmentationDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

for split in ["train", "val", "test", "test_official"]:
    ds = ClassificationDataset(split)
    x, y, iid = ds[0]
    dl = DataLoader(ds, batch_size=32, shuffle=(split == "train"), num_workers=0)
    t0 = time.time(); xb, yb, ids = next(iter(dl)); dt = time.time() - t0
    xb = xb.to(dev)
    assert xb.shape[1:] == (3, 224, 224), xb.shape
    assert xb.dtype == torch.float32 and -3 < xb.min() < 0 < xb.max() < 3, "normalização suspeita"
    print(f"[cls/{split:13s}] n={len(ds):5d} batch={tuple(xb.shape)} y∈[{yb.min()},{yb.max()}] "
          f"1º batch {dt:.2f}s | classes: {ds.class_counts().to_dict()}")

for split in ["train", "val", "test"]:
    ds = SegmentationDataset(split)
    dl = DataLoader(ds, batch_size=16, shuffle=(split == "train"), num_workers=0)
    xb, mb, ids = next(iter(dl)); xb, mb = xb.to(dev), mb.to(dev)
    assert xb.shape[1:] == (3, 256, 256) and mb.shape[1:] == (1, 256, 256), (xb.shape, mb.shape)
    assert set(torch.unique(mb).tolist()) <= {0.0, 1.0}, "máscara não binária"
    print(f"[seg/{split:5s}] n={len(ds):4d} batch={tuple(xb.shape)} mask={tuple(mb.shape)} "
          f"fração lesão média={mb.mean().item():.3f}")

print("\nOK — pipeline de dados funcional.")
