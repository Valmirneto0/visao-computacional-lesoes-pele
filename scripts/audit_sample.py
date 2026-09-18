"""Amostra para auditoria manual de artefatos (M2).

Gera reports/m2_dados/auditoria/:
  - sheet_XX.png   : grades 4x5 com 20 imagens numeradas (#1 … #200)
  - rotulos_manuais.csv : uma linha por imagem, colunas de artefato para preencher com 0/1

Amostra estratificada e reproduzível (seed 42): 40 MEL, 40 NV, 20 de cada outra classe (Task 3, treino)
+ 20 da Task 1 = 200 imagens. Uso: python scripts/audit_sample.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_PER = {"MEL": 40, "NV": 40, "BCC": 20, "AKIEC": 20, "BKL": 20, "DF": 20, "VASC": 20}
N_SEG = 20
COLS = ["pelo", "regua", "vinheta", "bolha", "caneta", "outro"]
OUT = ROOT / "reports/m2_dados/auditoria"


def main():
    data = yaml.safe_load(open(ROOT / "configs/data.yaml"))
    pre = yaml.safe_load(open(ROOT / "configs/preprocess.yaml"))
    man = pd.read_csv(ROOT / pre["output"]["manifest"])
    gt = pd.read_csv(ROOT / data["task3_labels"]); gt.columns = [c.strip() for c in gt.columns]
    gt["dx"] = gt[data["classes"]].idxmax(axis=1)
    cls = man[(man.task == "cls") & (man.source == "train")].merge(gt[["image", "dx"]], left_on="id", right_on="image")
    seg = man[man.task == "seg"]

    rng = np.random.default_rng(SEED)
    parts = []
    for c, n in N_PER.items():
        sub = cls[cls.dx == c]
        parts.append(sub.iloc[rng.choice(len(sub), size=min(n, len(sub)), replace=False)])
    parts.append(seg.iloc[rng.choice(len(seg), size=min(N_SEG, len(seg)), replace=False)].assign(dx="SEG"))
    s = pd.concat(parts).reset_index(drop=True)
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)  # embaralha para não rotular por bloco de classe
    s["n"] = np.arange(1, len(s) + 1)

    OUT.mkdir(parents=True, exist_ok=True)
    tw, th, cols, rows_ = 300, 225, 5, 4
    per = cols * rows_
    for k in range(0, len(s), per):
        chunk = s.iloc[k:k + per]
        sheet = np.full((rows_ * (th + 30), cols * tw, 3), 20, np.uint8)
        for j, (_, r) in enumerate(chunk.iterrows()):
            im = cv2.imread(str(ROOT / r.path)); im = cv2.resize(im, (tw, th))
            y0, x0 = (j // cols) * (th + 30), (j % cols) * tw
            sheet[y0 + 30:y0 + 30 + th, x0:x0 + tw] = im
            cv2.putText(sheet, f"#{r.n}  {r.id}", (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"sheet_{k // per + 1:02d}.png"), sheet)
        s.loc[chunk.index, "sheet"] = k // per + 1

    s["sheet"] = s["sheet"].astype(int)
    csv_path = OUT / "rotulos_manuais.csv"
    if csv_path.exists():
        print(f"{csv_path.relative_to(ROOT)} já existe — não sobrescrevo (pode ter rótulos preenchidos).")
    else:
        out = s[["n", "sheet", "id", "task", "dx"]].copy()
        for c in COLS: out[c] = ""
        out["obs"] = ""
        out.to_csv(csv_path, index=False)
        print(f"planilha: {csv_path.relative_to(ROOT)}")
    print(f"{len(s)} imagens em {int(s.sheet.max())} grades → {OUT.relative_to(ROOT)}")
    print(s.dx.value_counts().to_string())


if __name__ == "__main__":
    main()
