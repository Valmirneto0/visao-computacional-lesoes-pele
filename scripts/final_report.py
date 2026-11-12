"""Comparação dos experimentos, escolha do modelo final, avaliação no teste e análise de erros (entrega final).

Uso (depois de treinar os 3 experimentos):
    python scripts/final_report.py

1. Compara baseline × experimentos na VALIDAÇÃO (o teste não participa da escolha).
2. Escolhe o modelo final = maior macro-F1 na validação.
3. Só então avalia o modelo final no teste interno e no teste oficial ISIC 2018 (holdout externo).
4. Gera curvas, matrizes de confusão, galerias de acertos/erros, a análise de erros e a análise de confiança.
Saídas: reports/final/ (CSV/JSON) e reports/figures/2x_*.png.
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
import seaborn as sns
import yaml
from torch.utils.data import DataLoader

from skinlesion.data import ROOT, ClassificationDataset
from skinlesion.engine import get_device, load_checkpoint, predict
from skinlesion.metrics import compute_metrics

RUNS = ["baseline", "exp1_focal", "exp2_scratch"]
LABEL = {"baseline": "Baseline\n(ImageNet + CE pond.)", "exp1_focal": "Exp. 1\n(focal loss)",
         "exp2_scratch": "Exp. 2\n(do zero)"}
OUT = ROOT / "reports/final"; FIG = ROOT / "reports/figures"


def comparar():
    rows, hist = [], {}
    for r in RUNS:
        d = OUT / "runs" / r
        if not (d / "metrics_val.json").exists():
            print(f"  ! {r} ainda não treinado — pulando"); continue
        m = json.load(open(d / "metrics_val.json")); info = json.load(open(d / "run_info.json"))
        hist[r] = pd.read_csv(d / "history.csv")
        rows.append({"experimento": r, "melhor_epoca": info["best_epoch"], "epocas": info["epochs"],
                     "val_acuracia": m["accuracy"], "val_acuracia_balanceada": m["balanced_accuracy"],
                     "val_macro_precisao": m["macro_precision"], "val_macro_revocacao": m["macro_recall"],
                     "val_macro_f1": m["macro_f1"], "val_revocacao_MEL": m["per_class"]["MEL"]["recall"],
                     "val_f1_MEL": m["per_class"]["MEL"]["f1"], "minutos": info["minutes"]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "comparacao_experimentos.csv", index=False)
    if "baseline" in set(t.experimento):
        b = t.set_index("experimento").loc["baseline"]
        t["delta_macro_f1_vs_baseline"] = t.val_macro_f1 - b.val_macro_f1
        t["delta_rev_MEL_vs_baseline"] = t.val_revocacao_MEL - b.val_revocacao_MEL
        t.to_csv(OUT / "comparacao_experimentos.csv", index=False)
    print(t.round(4).to_string(index=False))
    return t, hist


def fig_curvas(hist):
    for r, h in hist.items():
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(h.epoch, h.train_loss, label="treino"); ax[0].plot(h.epoch, h.val_loss, label="validação")
        ax[0].set_title("Perda"); ax[0].legend()
        ax[1].plot(h.epoch, h.train_acc, label="treino"); ax[1].plot(h.epoch, h.val_acc, label="validação")
        ax[1].set_title("Acurácia"); ax[1].legend()
        ax[2].plot(h.epoch, h.val_macro_f1, label="macro-F1 val"); ax[2].plot(h.epoch, h.val_recall_MEL, label="revocação MEL val")
        best = h.val_macro_f1.idxmax(); ax[2].axvline(h.epoch[best], ls="--", c="gray", lw=1)
        ax[2].set_title("Validação"); ax[2].legend()
        for a in ax: a.set_xlabel("época"); a.grid(alpha=.3)
        fig.suptitle(f"Curvas de treinamento e validação — {r}"); fig.tight_layout()
        fig.savefig(FIG / f"20_curvas_{r}.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for r, h in hist.items():
        ax[0].plot(h.epoch, h.val_loss, label=r); ax[1].plot(h.epoch, h.val_macro_f1, label=r)
    ax[0].set_title("Perda na validação"); ax[1].set_title("Macro-F1 na validação")
    for a in ax: a.legend(); a.grid(alpha=.3); a.set_xlabel("época")
    fig.tight_layout(); fig.savefig(FIG / "21_curvas_comparadas.png", dpi=150); plt.close(fig)


def fig_barras(t):
    cols = ["val_acuracia", "val_acuracia_balanceada", "val_macro_f1", "val_revocacao_MEL"]
    nomes = ["Acurácia", "Acurácia balanceada", "Macro-F1", "Revocação MEL"]
    d = t.set_index("experimento")[cols]; d.columns = nomes
    ax = d.T.plot.bar(figsize=(10, 4.5), rot=0, width=.75)
    ax.set_ylim(0, 1); ax.set_title("Comparação na validação (mesmo split, seed e nº de épocas)"); ax.grid(axis="y", alpha=.3)
    for c in ax.containers: ax.bar_label(c, fmt="%.2f", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "22_comparacao_experimentos.png", dpi=150); plt.close()


def avaliar(ck_path, split, dev, nw):
    net, ck = load_checkpoint(ck_path, dev)
    ds = ClassificationDataset(split)
    ids, y, p, _ = predict(net, DataLoader(ds, batch_size=64, num_workers=nw), dev)
    m = compute_metrics(y, p.argmax(1), ds.classes)
    df = pd.DataFrame(p, columns=[f"p_{c}" for c in ds.classes]); df.insert(0, "id", ids)
    df.insert(1, "y", y); df.insert(2, "pred", p.argmax(1)); df["conf"] = p.max(1)
    df["path"] = ds.m.set_index("id").loc[ids, "path"].values
    return m, df, ds.classes


def fig_cm(m, classes, titulo, arq):
    cm = np.array(m["confusion_matrix"]); cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax[0], cbar=False)
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax[1], vmin=0, vmax=1)
    ax[0].set_title("Contagens"); ax[1].set_title("Normalizada por linha (= revocação na diagonal)")
    for a in ax: a.set_xlabel("previsto"); a.set_ylabel("real")
    fig.suptitle(titulo); fig.tight_layout(); fig.savefig(FIG / arq, dpi=150); plt.close(fig)


def galeria(df, classes, corretas, arq, n=14):
    rng = np.random.default_rng(42)
    if corretas:   # 2 por classe, sorteadas entre as corretas (não só as mais fáceis)
        sel = [g.sample(min(2, len(g)), random_state=42) for _, g in df[df.y == df.pred].groupby("y")]
        sel = pd.concat(sel).head(n); titulo = "Exemplos de previsões CORRETAS (teste interno)"
    else:          # erros mais confiantes primeiro: são os mais informativos
        sel = df[df.y != df.pred].sort_values("conf", ascending=False).head(n); titulo = "Previsões INCORRETAS mais confiantes (teste interno)"
    cols = 7; rows = int(np.ceil(len(sel) / cols)) or 1
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6)); axs = np.atleast_1d(axs).ravel()
    for a in axs: a.axis("off")
    for a, (_, r) in zip(axs, sel.iterrows()):
        img = cv2.cvtColor(cv2.imread(str(ROOT / r.path)), cv2.COLOR_BGR2RGB); a.imshow(img)
        a.set_title(f"real {classes[r.y]} | prev {classes[r.pred]}\nconf {r.conf:.2f}", fontsize=8,
                    color="green" if r.y == r.pred else "red")
    fig.suptitle(titulo); fig.tight_layout(); fig.savefig(FIG / arq, dpi=130); plt.close(fig)


def analise_erros(df, classes):
    """Causas dos erros, cruzando com o que a M2 mediu: confusões, artefato de vinheta, tipo de confirmação, confiança."""
    lim = yaml.safe_load(open(ROOT / "configs/artifacts.yaml"))["limiares"]["vignette"]
    art = pd.read_csv(ROOT / "reports/m2_dados/artefatos_scores.csv")[["id", "vignette"]]
    grp = pd.read_csv(ROOT / "data/raw/ISIC2018_Task3_Training_LesionGroupings.csv")
    d = df.merge(art, on="id", how="left").merge(grp[["image", "diagnosis_confirm_type"]], left_on="id", right_on="image", how="left")
    d["acerto"] = d.y == d.pred; d["vinheta"] = d.vignette < lim
    d["histopatologia"] = d.diagnosis_confirm_type.eq("histopathology")
    err = d[~d.acerto]
    pares = (err.assign(real=err.y.map(lambda i: classes[i]), prev=err.pred.map(lambda i: classes[i]))
             .groupby(["real", "prev"]).size().sort_values(ascending=False))
    def sub(mask):
        s = d[mask]; return compute_metrics(s.y, s.pred, classes) if len(s) else None
    out = {"n_teste": len(d), "n_erros": int((~d.acerto).sum()), "taxa_erro": float((~d.acerto).mean()),
           "top_confusoes": [{"real": a, "previsto": b, "n": int(n), "frac_dos_erros": float(n / len(err))}
                             for (a, b), n in pares.head(8).items()],
           "confianca_media_acertos": float(d[d.acerto].conf.mean()), "confianca_media_erros": float(err.conf.mean()),
           "erros_com_conf_maior_0.9": int((err.conf > .9).sum()),
           "por_vinheta": {}, "por_confirmacao": {}}
    for nome, mk in [("com_vinheta", d.vinheta), ("sem_vinheta", ~d.vinheta)]:
        s = sub(mk); out["por_vinheta"][nome] = None if s is None else {
            "n": s["n"], "acuracia": s["accuracy"], "macro_f1": s["macro_f1"],
            "revocacao_MEL": s["per_class"]["MEL"]["recall"], "suporte_MEL": s["per_class"]["MEL"]["support"],
            "precisao_MEL": s["per_class"]["MEL"]["precision"]}
    for nome, mk in [("histopatologia", d.histopatologia), ("outra_confirmacao", ~d.histopatologia)]:
        s = sub(mk); out["por_confirmacao"][nome] = None if s is None else {
            "n": s["n"], "acuracia": s["accuracy"], "macro_f1": s["macro_f1"]}
    # MEL previsto em imagens com vinheta que NÃO são MEL → indício de atalho
    nao_mel_vinheta = d[(d.y != 0) & d.vinheta]; nao_mel_sem = d[(d.y != 0) & ~d.vinheta]
    out["falso_MEL_taxa_com_vinheta"] = float((nao_mel_vinheta.pred == 0).mean()) if len(nao_mel_vinheta) else None
    out["falso_MEL_taxa_sem_vinheta"] = float((nao_mel_sem.pred == 0).mean()) if len(nao_mel_sem) else None
    # figura: confiança de acertos × erros
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d[d.acerto].conf, bins=20, range=(0, 1), alpha=.6, label="acertos", density=True)
    ax.hist(err.conf, bins=20, range=(0, 1), alpha=.6, label="erros", density=True)
    ax.set_xlabel("confiança (prob. máxima)"); ax.set_ylabel("densidade"); ax.legend()
    ax.set_title("Confiança do modelo final — teste interno"); fig.tight_layout()
    fig.savefig(FIG / "27_confianca_acertos_erros.png", dpi=150); plt.close(fig)
    d.drop(columns=["image"]).to_csv(OUT / "preds_test_detalhado.csv", index=False)
    return out


def fig_f1_classe(mt, mo, classes):
    d = pd.DataFrame({"Teste interno": [mt["per_class"][c]["f1"] for c in classes],
                      "Teste oficial ISIC": [mo["per_class"][c]["f1"] for c in classes]}, index=classes)
    ax = d.plot.bar(figsize=(9, 4), rot=0); ax.set_ylim(0, 1); ax.set_ylabel("F1"); ax.grid(axis="y", alpha=.3)
    ax.set_title("F1 por classe — modelo final")
    for c in ax.containers: ax.bar_label(c, fmt="%.2f", fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "28_f1_por_classe.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device"); ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--final", help="força o experimento final (padrão: maior macro-F1 na validação)")
    a = ap.parse_args(); dev = get_device(a.device); FIG.mkdir(parents=True, exist_ok=True)

    print("== 1. Comparação na validação"); t, hist = comparar(); fig_curvas(hist); fig_barras(t)
    final = a.final or t.sort_values("val_macro_f1", ascending=False).experimento.iloc[0]
    print(f"== 2. Modelo final: {final}")
    ck = ROOT / "models" / f"{final}.pt"

    print("== 3. Avaliação final (teste interno e teste oficial ISIC 2018)")
    mt, dft, classes = avaliar(ck, "test", dev, a.num_workers)
    mo, dfo, _ = avaliar(ck, "test_official", dev, a.num_workers)
    json.dump(mt, open(OUT / "metrics_test.json", "w"), indent=2); json.dump(mo, open(OUT / "metrics_test_official.json", "w"), indent=2)
    dft.to_csv(OUT / "preds_test.csv", index=False); dfo.to_csv(OUT / "preds_test_official.csv", index=False)
    fig_cm(mt, classes, f"Matriz de confusão — teste interno (n={mt['n']}) — {final}", "23_matriz_confusao_teste.png")
    fig_cm(mo, classes, f"Matriz de confusão — teste oficial ISIC 2018 (n={mo['n']}) — {final}", "24_matriz_confusao_teste_oficial.png")
    fig_f1_classe(mt, mo, classes)

    print("== 4. Acertos, erros e causas")
    galeria(dft, classes, True, "25_acertos.png"); galeria(dft, classes, False, "26_erros.png")
    ae = analise_erros(dft, classes); json.dump(ae, open(OUT / "analise_erros.json", "w"), indent=2, ensure_ascii=False)
    from skinlesion.confianca import analise_confianca   # o modelo sabe quando erra? (pergunta da M1)
    conf = analise_confianca(ROOT)

    sel = t.set_index("experimento").loc[final]
    resumo = {"modelo_final": final, "checkpoint": str(ck.relative_to(ROOT)),
              "criterio": "maior macro-F1 na validação (teste não usado na escolha)",
              "val_macro_f1": float(sel.val_macro_f1),
              "teste": {k: mt[k] for k in ["n", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]},
              "teste_oficial": {k: mo[k] for k in ["n", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]},
              "teste_revocacao_MEL": mt["per_class"]["MEL"]["recall"], "teste_oficial_revocacao_MEL": mo["per_class"]["MEL"]["recall"]}
    json.dump(resumo, open(OUT / "modelo_final.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps(ae, indent=2, ensure_ascii=False))
    print(json.dumps(conf, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
