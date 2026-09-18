"""Pré-processamento (M2): redimensiona ISIC 2018 uma única vez para data/processed/.

Uso (na raiz do projeto, venv ativo):
    python scripts/preprocess.py                # tudo
    python scripts/preprocess.py --task cls     # só classificação (Task 3 treino + teste oficial)
    python scripts/preprocess.py --task seg     # só segmentação (Task 1)
    python scripts/preprocess.py --limit 50     # amostra, para testar

Idempotente: pula arquivos já gerados. Gera data/processed/manifest_v1.csv.
"""
import argparse, csv, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
INTERP = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST, "cubic": cv2.INTER_CUBIC}


def load_cfg():
    data = yaml.safe_load(open(ROOT / "configs/data.yaml"))
    pre = yaml.safe_load(open(ROOT / "configs/preprocess.yaml"))
    return data, pre


def resize_short_side(img, short, interp):
    h, w = img.shape[:2]
    s = short / min(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp)


def process_cls(args):
    src, dst, short, interp, q = args
    if dst.exists():
        return src.stem, None, None, "skip"
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return src.stem, None, None, "erro_leitura"
    h, w = img.shape[:2]
    out = resize_short_side(img, short, interp)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return src.stem, w, h, "ok"


def process_seg(args):
    src_img, src_msk, dst_img, dst_msk, size, interp, minterp, comp = args
    if dst_img.exists() and dst_msk.exists():
        return src_img.stem, None, None, "skip"
    img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    msk = cv2.imread(str(src_msk), cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        return src_img.stem, None, None, "erro_leitura"
    if img.shape[:2] != msk.shape[:2]:
        return src_img.stem, None, None, "shape_diferente"
    h, w = img.shape[:2]
    img_o = cv2.resize(img, (size, size), interpolation=interp)
    msk_o = cv2.resize(msk, (size, size), interpolation=minterp)
    msk_o = np.where(msk_o > 127, 255, 0).astype(np.uint8)  # garante binária
    dst_img.parent.mkdir(parents=True, exist_ok=True); dst_msk.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_img), img_o, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(dst_msk), msk_o, [cv2.IMWRITE_PNG_COMPRESSION, comp])
    return src_img.stem, w, h, "ok"


def run(jobs, fn, workers, label):
    rows, t0, counts = [], time.time(), {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result()); counts[rows[-1][3]] = counts.get(rows[-1][3], 0) + 1
            if i % 500 == 0 or i == len(jobs):
                print(f"  {label}: {i}/{len(jobs)}  {time.time()-t0:.0f}s  {counts}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cls", "seg", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="processa só as N primeiras de cada fonte (teste)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    data, pre = load_cfg()
    out_root = ROOT / pre["output"]["root"]
    split = pd.read_csv(ROOT / "data/splits/split_v1.csv")
    split_map = dict(zip(split.image, split.split))
    manifest = []

    if a.task in ("cls", "all"):
        c = pre["classification"]
        for name, key in c["sources"].items():
            src_dir = ROOT / data[key]
            files = sorted(src_dir.glob("*.jpg"))[: a.limit]
            dst_dir = out_root / "cls" / name
            print(f"[cls/{name}] {len(files)} imagens de {src_dir.name} → {dst_dir.relative_to(ROOT)}")
            jobs = [(f, dst_dir / f.name, c["short_side"], INTERP[c["interpolation"]], c["jpeg_quality"]) for f in files]
            for iid, w, h, st in run(jobs, process_cls, a.workers, f"cls/{name}"):
                sp = split_map.get(iid, "test_official" if name == "test_official" else "NA")
                manifest.append(dict(id=iid, task="cls", source=name, split=sp, path=(dst_dir / f"{iid}.jpg").relative_to(ROOT).as_posix(), w_orig=w, h_orig=h, status=st))

    if a.task in ("seg", "all"):
        s = pre["segmentation"]
        img_dir, msk_dir = ROOT / data[s["sources"]["images"]], ROOT / data[s["sources"]["masks"]]
        files = sorted(img_dir.glob("*.jpg"))[: a.limit]
        dst_i, dst_m = out_root / "seg" / "images", out_root / "seg" / "masks"
        print(f"[seg] {len(files)} pares → {dst_i.relative_to(ROOT)}, {dst_m.relative_to(ROOT)}")
        jobs = [(f, msk_dir / f"{f.stem}_segmentation.png", dst_i / f.name, dst_m / f"{f.stem}.png",
                 s["size"], INTERP[s["interpolation"]], INTERP[s["mask_interpolation"]], s["png_compression"]) for f in files]
        for iid, w, h, st in run(jobs, process_seg, a.workers, "seg"):
            manifest.append(dict(id=iid, task="seg", source="task1", split=split_map.get(iid, "NA"), path=(dst_i / f"{iid}.jpg").relative_to(ROOT).as_posix(), w_orig=w, h_orig=h, status=st))

    # manifest: junta com o anterior se existir (execuções parciais), sem duplicar
    mpath = ROOT / pre["output"]["manifest"]
    new = pd.DataFrame(manifest)
    key = ["id", "task", "source"]
    if mpath.exists():
        old = pd.read_csv(mpath)
        # linhas 'skip' não têm w/h: mantém a versão antiga delas; as demais são substituídas
        skipped = new[new.status == "skip"].set_index(key).index
        new = new[new.status != "skip"]
        old_keep = old[old.set_index(key).index.isin(skipped) | ~old.set_index(key).index.isin(new.set_index(key).index)]
        new = pd.concat([old_keep, new])
    new["path"] = new.path.str.replace("\\", "/", regex=False)  # separador único, mesmo no Windows
    new = new.sort_values(["task", "source", "id"])
    mpath.parent.mkdir(parents=True, exist_ok=True); new.to_csv(mpath, index=False)
    print(f"\nmanifest: {mpath.relative_to(ROOT)}  ({len(new)} linhas)")
    print(new.groupby(["task", "source", "split"]).size().to_string())
    bad = new[~new.status.isin(["ok", "skip"])]
    if len(bad):
        print(f"\nATENÇÃO: {len(bad)} arquivos com problema:\n{bad.head(20).to_string()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
