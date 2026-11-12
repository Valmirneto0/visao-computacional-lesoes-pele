"""O modelo sabe quando está errando? (entrega final — terceira parte da pergunta da M1).

Usa só as previsões já salvas (coluna `conf` = probabilidade máxima do softmax, `acerto` = previsão correta).
- AUROC da confiança como detector de acerto: 0,5 = a confiança não distingue acerto de erro; 1,0 = separa perfeitamente.
- Acurácia e cobertura por limiar: se o modelo só responder quando conf ≥ t, quantas imagens ele cobre e quanto acerta.
Não depende de torch: roda a partir de reports/final/preds_test*.csv.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LIMIARES = [0.5, 0.7, 0.9]


def resumo(df):
    acerto = (df.y == df.pred) if "acerto" not in df else df.acerto.astype(bool)
    conf = df.conf.values
    out = {"n": int(len(df)), "acuracia": float(acerto.mean()),
           "auroc_confianca_acerto": float(roc_auc_score(acerto, conf)),
           "confianca_media_acertos": float(conf[acerto].mean()), "confianca_media_erros": float(conf[~acerto].mean()),
           "n_erros": int((~acerto).sum()), "erros_conf_maior_0.9": int(((~acerto) & (conf > 0.9)).sum()),
           "por_limiar": []}
    for t in LIMIARES:
        m = conf >= t
        out["por_limiar"].append({"limiar": t, "cobertura": float(m.mean()), "acuracia_coberta": float(acerto[m].mean()),
                                  "acuracia_abaixo": float(acerto[~m].mean()) if (~m).any() else None, "n_abaixo": int((~m).sum())})
    return out, acerto.values, conf


def figura(curvas, arq):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for nome, (acerto, conf) in curvas.items():
        ts = np.linspace(0, 0.99, 100)
        cob = [(conf >= t).mean() for t in ts]; acc = [acerto[conf >= t].mean() for t in ts]
        ax.plot(cob, acc, label=nome)
    ax.set_xlabel("cobertura (fração de imagens com confiança ≥ limiar)"); ax.set_ylabel("acurácia nas imagens cobertas")
    ax.set_title("Confiança × acurácia: o modelo sabe quando está errando?"); ax.grid(alpha=.3); ax.legend()
    ax.invert_xaxis(); fig.tight_layout(); fig.savefig(arq, dpi=150); plt.close(fig)


def analise_confianca(root):
    out_dir, fig_dir = root / "reports/final", root / "reports/figures"
    res, curvas = {}, {}
    for nome, arq in [("teste_interno", "preds_test.csv"), ("teste_oficial", "preds_test_official.csv")]:
        df = pd.read_csv(out_dir / arq)
        res[nome], a, c = resumo(df); curvas[nome.replace("_", " ")] = (a, c)
    json.dump(res, open(out_dir / "analise_confianca.json", "w"), indent=2, ensure_ascii=False)
    figura(curvas, fig_dir / "30_confianca_cobertura.png")
    return res
