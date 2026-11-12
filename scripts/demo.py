"""Demonstração do fluxo completo: imagem → pré-processamento → modelo → resultado.

Uso:
    python scripts/demo.py                                  # 7 imagens (1 por classe) sorteadas do teste oficial ISIC 2018 (nunca usadas no treino)
    python scripts/demo.py --images caminho/a.jpg caminho/b.jpg
    python scripts/demo.py --model models/baseline.pt

Parte da imagem ORIGINAL (qualquer tamanho), não da versão em data/processed.
Pré-processamento idêntico ao do treino: lado menor → 256 (INTER_AREA, como scripts/preprocess.py),
recompressão JPEG q95 (como os arquivos processados), crop central 224 e normalização ImageNet (configs/transforms.yaml).
Saídas: reports/figures/29_demo.png e reports/final/demo_resultados.csv.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from skinlesion.data import ROOT, build_transforms, load_yaml
from skinlesion.engine import get_device, load_checkpoint

NOMES = {"MEL": "melanoma", "NV": "nevo melanocítico", "BCC": "carcinoma basocelular", "AKIEC": "queratose actínica / Bowen",
         "BKL": "lesão queratósica benigna", "DF": "dermatofibroma", "VASC": "lesão vascular"}


def preprocessar(path, tf):
    """imagem original (BGR, qualquer tamanho) → tensor [1,3,224,224] + imagem RGB 224 para exibir."""
    pre = load_yaml("configs/preprocess.yaml")["classification"]
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]; s = pre["short_side"] / min(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, pre["jpeg_quality"]])[1], cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = tf(image=rgb)["image"].unsqueeze(0)
    return x, (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*")
    ap.add_argument("--model", help="padrão: o modelo final de reports/final/modelo_final.json")
    ap.add_argument("--n", type=int, default=7); ap.add_argument("--device")
    a = ap.parse_args(); dev = get_device(a.device)

    ck = a.model or json.load(open(ROOT / "reports/final/modelo_final.json"))["checkpoint"]
    net, info = load_checkpoint(ROOT / str(ck).replace("\\", "/"), dev); classes = info["classes"]   # "\\" → "/": caminho salvo no Windows vale em qualquer SO
    tf = build_transforms("classification", "test")       # transformações de avaliação (sem augmentation)

    gt = {}
    if a.images:
        paths = [Path(p) for p in a.images]
    else:   # teste oficial ISIC 2018: holdout externo, nunca visto no treino nem na seleção do modelo
        dcfg = load_yaml("configs/data.yaml")
        g = pd.read_csv(ROOT / dcfg["task3_test_labels"]); g["dx"] = g[classes].idxmax(axis=1); gt = dict(zip(g.image, g.dx))
        # sorteio estratificado leve: 1 por classe até completar n (mostra classes além de NV)
        rng = np.random.default_rng(7); escolha = []
        for c in classes:
            ids = g[g.dx == c].image.values
            escolha.append(rng.choice(ids))
        paths = [ROOT / dcfg["task3_test_images"] / f"{i}.jpg" for i in escolha[:a.n]]

    rows = []
    for p in paths:
        x, (w, h) = preprocessar(p, tf)
        with torch.no_grad():
            prob = torch.softmax(net(x.to(dev)), 1)[0].cpu().numpy()
        top = np.argsort(prob)[::-1][:3]
        r = {"imagem": p.stem, "tamanho_original": f"{w}x{h}", "previsto": classes[top[0]], "confianca": float(prob[top[0]]),
             "top3": " | ".join(f"{classes[i]} {prob[i]:.2f}" for i in top), "real": gt.get(p.stem, "?")}
        rows.append(r); print(f"{r['imagem']}: {r['previsto']} ({NOMES[r['previsto']]}) conf {r['confianca']:.2f} "
                              f"| top-3: {r['top3']} | real: {r['real']}")
    df = pd.DataFrame(rows); (ROOT / "reports/final").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "reports/final/demo_resultados.csv", index=False)

    n = len(paths); fig, axs = plt.subplots(2, n, figsize=(2.9 * n, 6.2), gridspec_kw={"height_ratios": [3, 2]})
    axs = np.array(axs).reshape(2, n)
    for j, (p, r) in enumerate(zip(paths, rows)):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB); axs[0, j].imshow(img); axs[0, j].axis("off")
        ok = r["real"] == "?" or r["real"] == r["previsto"]
        axs[0, j].set_title(f"{r['imagem']}\nreal: {r['real']} | prev: {r['previsto']}", fontsize=8, color="green" if ok else "red")
        x, _ = preprocessar(p, tf)
        with torch.no_grad():
            prob = torch.softmax(net(x.to(dev)), 1)[0].cpu().numpy()
        axs[1, j].barh(classes, prob, color=["#c0392b" if c == r["previsto"] else "#95a5a6" for c in classes])
        axs[1, j].set_xlim(0, 1); axs[1, j].invert_yaxis(); axs[1, j].tick_params(labelsize=7)
    fig.suptitle("Demonstração: imagem original → pré-processamento → ResNet-18 → probabilidades (7 classes)")
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/29_demo.png", dpi=140); plt.close(fig)
    print("→ reports/figures/29_demo.png | reports/final/demo_resultados.csv")


if __name__ == "__main__":
    main()
