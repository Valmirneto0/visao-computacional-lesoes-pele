"""Treina um experimento de classificação (entrega final).

Uso (raiz do projeto, venv ativo):
    python scripts/train.py --config configs/experiments/baseline.yaml
    python scripts/train.py --config configs/experiments/exp1_focal.yaml
    python scripts/train.py --config configs/experiments/exp2_scratch.yaml
    python scripts/train.py --config ... --epochs 1 --limit 64     # teste rápido do fluxo

Usa SÓ treino e validação. A época salva é a de maior macro-F1 na validação.
Saídas: models/<name>.pt e reports/final/runs/<name>/ (history.csv, metrics_val.json, preds_val.csv, run_info.json).
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from skinlesion.data import ROOT, ClassificationDataset
from skinlesion.engine import get_device, git_hash, load_cfg, predict, set_seed, train_one_epoch
from skinlesion.losses import build_loss
from skinlesion.metrics import compute_metrics
from skinlesion.models import build_model, count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, help="sobrescreve o YAML (teste rápido)")
    ap.add_argument("--limit", type=int, help="usa só N imagens de cada split (teste rápido)")
    ap.add_argument("--device")
    ap.add_argument("--num-workers", type=int)
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    if a.epochs: cfg["epochs"] = a.epochs
    if a.num_workers is not None: cfg["num_workers"] = a.num_workers
    name = cfg["name"]; set_seed(cfg["seed"]); dev = get_device(a.device)
    out = ROOT / "reports/final/runs" / name; out.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True); ck_path = ROOT / "models" / f"{name}.pt"

    tr, va = ClassificationDataset("train"), ClassificationDataset("val")
    classes = tr.classes
    tr_ds, va_ds = tr, va
    if a.limit:
        g = torch.Generator().manual_seed(cfg["seed"])
        tr_ds = Subset(tr, torch.randperm(len(tr), generator=g)[:a.limit].tolist())
        va_ds = Subset(va, torch.randperm(len(va), generator=g)[:a.limit].tolist())
    nw = cfg["num_workers"]
    g = torch.Generator().manual_seed(cfg["seed"])
    dl_tr = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, generator=g,
                       pin_memory=dev.type == "cuda", persistent_workers=nw > 0, drop_last=False)
    dl_va = DataLoader(va_ds, batch_size=cfg["batch_size"] * 2, shuffle=False, num_workers=nw,
                       pin_memory=dev.type == "cuda", persistent_workers=nw > 0)

    net = build_model(cfg["model"], len(classes)).to(dev)
    crit = build_loss(cfg["loss"], classes).to(dev)
    oc = cfg["optimizer"]
    opt = torch.optim.AdamW(net.parameters(), lr=float(oc["lr"]), weight_decay=float(oc["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"]) if cfg.get("scheduler") == "cosine" else None
    n_tot, n_tr = count_params(net)
    print(f"[{name}] device={dev} | treino={len(tr_ds)} val={len(va_ds)} | params={n_tot:,} | "
          f"entrada [B,3,224,224] → saída [B,{len(classes)}]", flush=True)

    hist, best, best_ep, t0 = [], -1.0, -1, time.time()
    for ep in range(1, cfg["epochs"] + 1):
        te = time.time()
        tr_loss, tr_acc = train_one_epoch(net, dl_tr, crit, opt, dev)
        _, yv, pv, va_loss = predict(net, dl_va, dev, crit)
        m = compute_metrics(yv, pv.argmax(1), classes)
        lr = opt.param_groups[0]["lr"]
        if sched: sched.step()
        row = {"epoch": ep, "lr": lr, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
               "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_balanced_acc": m["balanced_accuracy"],
               "val_recall_MEL": m["per_class"]["MEL"]["recall"], "seg": time.time() - te}
        hist.append(row); pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        flag = ""
        if m[cfg["selection_metric"]] > best:
            best, best_ep, flag = m[cfg["selection_metric"]], ep, " *"
            torch.save({"state_dict": net.state_dict(), "config": cfg, "classes": classes, "epoch": ep,
                        "val_metrics": m, "git": git_hash()}, ck_path)
        print(f"  ep {ep:02d}/{cfg['epochs']} | loss tr {tr_loss:.4f} val {va_loss:.4f} | acc tr {tr_acc:.3f} "
              f"val {m['accuracy']:.3f} | macroF1 val {m['macro_f1']:.3f} | recall MEL {row['val_recall_MEL']:.3f} "
              f"| {row['seg']:.0f}s{flag}", flush=True)

    # previsões de validação com o melhor checkpoint (base da comparação entre experimentos)
    ck = torch.load(ck_path, map_location=dev, weights_only=False); net.load_state_dict(ck["state_dict"])
    ids, yv, pv, _ = predict(net, dl_va, dev)
    m = compute_metrics(yv, pv.argmax(1), classes)
    json.dump(m, open(out / "metrics_val.json", "w"), indent=2)
    dfp = pd.DataFrame(pv, columns=[f"p_{c}" for c in classes]); dfp.insert(0, "id", ids)
    dfp.insert(1, "y", yv); dfp.insert(2, "pred", pv.argmax(1)); dfp.to_csv(out / "preds_val.csv", index=False)
    info = {"name": name, "config_file": a.config, "git": git_hash(), "device": str(dev),
            "torch": torch.__version__, "best_epoch": best_ep, "best_val_" + cfg["selection_metric"]: best,
            "epochs": cfg["epochs"], "params_total": n_tot, "params_trainable": n_tr, "minutes": (time.time() - t0) / 60,
            "n_train": len(tr_ds), "n_val": len(va_ds), "limit": a.limit, "checkpoint": str(ck_path.relative_to(ROOT))}
    json.dump(info, open(out / "run_info.json", "w"), indent=2)
    yaml.safe_dump(cfg, open(out / "config_usado.yaml", "w"), allow_unicode=True, sort_keys=False)
    print(f"[{name}] melhor época {best_ep}: macro-F1 val = {best:.4f} | {info['minutes']:.1f} min → {ck_path.name}")


if __name__ == "__main__":
    main()
