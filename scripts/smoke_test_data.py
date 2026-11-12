"""Smoke test do pipeline de dados (M2): carrega um batch de cada split de classificação e confere formas,
faixas de valores e se vai para a GPU. Roda em ~10 s. Uso: python scripts/smoke_test_data.py
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch.utils.data import DataLoader

from skinlesion.data import ClassificationDataset

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

print("\nOK — pipeline de dados funcional.")
