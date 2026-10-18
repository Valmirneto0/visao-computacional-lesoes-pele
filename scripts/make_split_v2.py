"""split_v2 (M2): corrige vazamento por lesão na Task 1 (segmentação).

Motivo (medido na auditoria M2): a Task 1 não tem lesion_id e o split_v1 dividiu por imagem. A busca por
pHash (Hamming ≤ 4, scripts/m2_evidencias.py) encontrou 50 pares; 14 grupos atravessavam splits e a
inspeção visual confirmou a mesma lesão em train/val/test (cópia quase idêntica e re-aquisições).

O que muda: só a Task 1. Pares com pHash ≤ 4 viram um grupo (união transitiva, critério conservador:
um falso positivo só junta duas imagens no mesmo split, sem custo de vazamento). Os grupos são
embaralhados com seed 42 e alocados em ordem até 70% / 15% / 15% das imagens.
Classificação (Task 3): idêntica ao split_v1 (StratifiedGroupKFold por lesion_id), copiada sem alteração.

Uso: python scripts/make_split_v2.py   (requer reports/m2_dados/hashes_v1.csv, gerado por m2_evidencias.py hash)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED, MAX_DIST, FRAC = 42, 4, {"train": 0.70, "val": 0.15}


def main():
    v1 = pd.read_csv(ROOT / "data/splits/split_v1.csv")
    h = pd.read_csv(ROOT / "reports/m2_dados/hashes_v1.csv")
    seg = v1[v1.task == "seg"][["image"]].merge(h[["id", "phash"]], left_on="image", right_on="id").sort_values("image").reset_index(drop=True)
    nib = np.array([[int(c, 16) for c in p] for p in seg.phash], dtype=np.uint8)
    pop = np.array([bin(i).count("1") for i in range(16)], dtype=np.uint8)
    parent = list(range(len(seg)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    n_pairs = 0
    for i in range(len(seg) - 1):
        d = pop[nib[i] ^ nib[i + 1:]].sum(1)
        for j in np.where(d <= MAX_DIST)[0]:
            parent[find(i)] = find(i + 1 + j); n_pairs += 1
    roots = [find(i) for i in range(len(seg))]
    seg["group"] = [seg.image[r] for r in roots]
    sizes = seg.group.value_counts()
    seg["lesion_id"] = np.where(seg.group.map(sizes) > 1, "T1G_" + seg.group.str.replace("ISIC_", ""), seg.image)

    groups = seg.lesion_id.unique()
    rng = np.random.default_rng(SEED); rng.shuffle(groups)
    gsize = seg.lesion_id.value_counts()
    n, acc, assign = len(seg), 0, {}
    lim_train, lim_val = FRAC["train"] * n, (FRAC["train"] + FRAC["val"]) * n
    for g in groups:
        assign[g] = "train" if acc < lim_train else ("val" if acc < lim_val else "test")
        acc += gsize[g]
    seg["split"] = seg.lesion_id.map(assign)
    assert (seg.groupby("lesion_id").split.nunique() == 1).all()

    out = pd.concat([v1[v1.task == "cls"], seg[["image", "lesion_id", "split"]].assign(dx=np.nan, task="seg")])[v1.columns]
    out.to_csv(ROOT / "data/splits/split_v2.csv", index=False)
    print(f"pares pHash≤{MAX_DIST}: {n_pairs} | grupos multi-imagem: {(sizes > 1).sum()} ({sizes[sizes > 1].sum()} imagens)")
    print("seg v1:", v1[v1.task == "seg"].split.value_counts().to_dict(), "→ v2:", seg.split.value_counts().to_dict())
    print("cls inalterado:", out[out.task == "cls"].equals(v1[v1.task == "cls"]))


if __name__ == "__main__":
    main()
