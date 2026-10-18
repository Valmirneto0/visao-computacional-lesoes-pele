"""Evidências da auditoria M2 em modo script (equivalente ao notebooks/02_auditoria.ipynb, sem torch).

Etapas (cada uma grava cache em reports/m2_dados/ e pode ser rodada separadamente):
    python scripts/m2_evidencias.py hash      # MD5 dos arquivos BRUTOS, modo/dimensão do bruto, pHash do processado
    python scripts/m2_evidencias.py scores    # scores de artefato (src/skinlesion/artifacts.py) + estatísticas das máscaras
    python scripts/m2_evidencias.py analyze   # duplicatas, limiares vs rótulos manuais, qui-quadrado, pesos, vazamento,
                                              # figuras e manifestos (data/splits/train|validation|test|test_official.csv)
    python scripts/m2_evidencias.py figures   # figuras 08b, 09b, 10 (vinheta detector×manual, augmentations com contorno, distribuição)
    python scripts/m2_evidencias.py all

Saídas: reports/m2_dados/evidencias_m2_script.json, configs/artifacts.yaml, data/splits/class_weights_v1.json,
data/splits/manifest_m2_v1.csv (+ train/validation/test/test_official.csv), reports/figures/08b, 09b, 10 (nomes distintos dos do notebook 02, que grava evidencias_m2.json e artefatos_scores.csv)
Convenção: medido / derivado / inferido, como nos notebooks.
"""
import hashlib, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from skinlesion.artifacts import all_scores  # noqa: E402

DATA = yaml.safe_load(open(ROOT / "configs/data.yaml", encoding="utf-8"))
PRE = yaml.safe_load(open(ROOT / "configs/preprocess.yaml", encoding="utf-8"))
TR = yaml.safe_load(open(ROOT / "configs/transforms.yaml", encoding="utf-8"))
CLASSES, SEED = DATA["classes"], DATA["seed"]
OUT = ROOT / "reports/m2_dados"; FIG = ROOT / "reports/figures"
SPLIT_NAME = {"train": "train", "val": "validation", "test": "test", "test_official": "holdout"}
LICENSE = {"cls": "CC BY-NC 4.0", "seg": "CC0 1.0"}


def manifest():
    m = pd.read_csv(ROOT / PRE["output"]["manifest"])
    m["path"] = m.path.str.replace("\\", "/", regex=False)
    raw = {("cls", "train"): DATA["task3_images"], ("cls", "test_official"): DATA["task3_test_images"], ("seg", "task1"): DATA["task1_images"]}
    m["raw_path"] = [f"{raw[(t, s)]}/{i}.jpg" for t, s, i in zip(m.task, m.source, m.id)]
    m["raw_mask"] = np.where(m.task == "seg", DATA["task1_masks"] + "/" + m.id + "_segmentation.png", "")
    m["mask_path"] = np.where(m.task == "seg", "data/processed/seg/masks/" + m.id + ".png", "")
    sp = pd.read_csv(ROOT / DATA["split_file"])
    m["split"] = np.where(m.source == "test_official", "test_official", m.id.map(dict(zip(sp.image, sp.split))))
    m["seg_group"] = m.id.map(dict(zip(sp[sp.task == "seg"].image, sp[sp.task == "seg"].lesion_id)))
    return m


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _hash_row(r):
    import imagehash
    rid, rawp, rawm, proc = r
    out = dict(id=rid)
    try:
        out["md5_raw"] = md5(ROOT / rawp)
        with Image.open(ROOT / rawp) as im:
            out.update(raw_mode=im.mode, raw_w=im.size[0], raw_h=im.size[1], raw_format=im.format)
        out["md5_mask_raw"] = md5(ROOT / rawm) if rawm else ""
        out["phash"] = str(imagehash.phash(Image.open(ROOT / proc).convert("RGB"), hash_size=8))
        out["hash_status"] = "ok"
    except Exception as e:  # noqa: BLE001
        out["hash_status"] = f"erro: {type(e).__name__}"
    return out


def stage_hash():
    m = manifest(); cache = OUT / "hashes_v1.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["id", "hash_status"])
    done = done[done.hash_status.notna()]  # descarta linha truncada de execução interrompida
    todo = m[~m.id.isin(done.id)]
    print(f"hash: {len(done)} em cache, {len(todo)} a calcular", flush=True)
    rows, t0 = [], time.time()
    with Pool(8) as p:
        for i, r in enumerate(p.imap_unordered(_hash_row, list(zip(todo.id, todo.raw_path, todo.raw_mask, todo.path)), chunksize=8), 1):
            rows.append(r)
            if i % 200 == 0 or i == len(todo):
                tmp = cache.with_suffix(".tmp"); pd.concat([done, pd.DataFrame(rows)]).to_csv(tmp, index=False); tmp.replace(cache)
                print(f"  {i}/{len(todo)} {time.time()-t0:.0f}s", flush=True)


def _score_row(r):
    rid, task, proc, mpath, rawm = r
    img = cv2.imread(str(ROOT / proc))
    out = dict(id=rid, task=task, proc_h=img.shape[0], proc_w=img.shape[1], proc_c=img.shape[2], **all_scores(img))
    if task == "seg":
        mk = cv2.imread(str(ROOT / mpath), cv2.IMREAD_GRAYSCALE)
        b = mk > 127
        out.update(mask_area=float(b.mean()), mask_touch_border=bool(b[0].any() or b[-1].any() or b[:, 0].any() or b[:, -1].any()),
                   mask_proc_binary=bool(set(np.unique(mk).tolist()) <= {0, 255}),
                   mask_components=int(cv2.connectedComponents(b.astype(np.uint8))[0] - 1))
        mr = cv2.imread(str(ROOT / rawm), cv2.IMREAD_GRAYSCALE)
        out["mask_raw_binary"] = bool(set(np.unique(mr).tolist()) <= {0, 255})
        out["mask_raw_shape"] = f"{mr.shape[1]}x{mr.shape[0]}"
    return out


def stage_scores():
    m = manifest(); cache = OUT / "artefatos_scores_script.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["id"])
    todo = m[~m.id.isin(done.id)]
    print(f"scores: {len(done)} em cache, {len(todo)} a calcular", flush=True)
    rows, t0 = [], time.time()
    with Pool(4) as p:
        for i, r in enumerate(p.imap_unordered(_score_row, list(zip(todo.id, todo.task, todo.path, todo.mask_path, todo.raw_mask)), chunksize=16), 1):
            rows.append(r)
            if i % 500 == 0 or i == len(todo):
                tmp = cache.with_suffix(".tmp"); pd.concat([done, pd.DataFrame(rows)]).to_csv(tmp, index=False); tmp.replace(cache)
                print(f"  {i}/{len(todo)} {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ análise
def labels():
    gt = pd.read_csv(ROOT / DATA["task3_labels"]); gt.columns = [c.strip() for c in gt.columns]
    gtt = pd.read_csv(ROOT / DATA["task3_test_labels"]); gtt.columns = [c.strip() for c in gtt.columns]
    for g in (gt, gtt):
        g["dx"] = g[CLASSES].idxmax(axis=1); g["n_pos"] = g[CLASSES].sum(axis=1)
    return pd.concat([gt[["image", "dx", "n_pos"]], gtt[["image", "dx", "n_pos"]]])


def near_pairs(df, max_dist=4):
    bits = np.array([[int(c, 16) for c in h] for h in df.phash], dtype=np.uint8)  # 16 nibbles
    pop = np.array([bin(i).count("1") for i in range(16)], dtype=np.uint8)
    pares = []
    for i in range(len(df) - 1):
        d = pop[bits[i] ^ bits[i + 1:]].sum(1)
        for j in np.where(d <= max_dist)[0]:
            pares.append((i, i + 1 + j, int(d[j])))
    return pd.DataFrame(pares, columns=["a", "b", "dist"])


def pixel_check(pa, pb):
    a = cv2.resize(cv2.imread(str(ROOT / pa)), (128, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    b0 = cv2.resize(cv2.imread(str(ROOT / pb)), (128, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    best = (1e9, 0.0)
    for f in (None, 0, 1, -1):  # aceita espelhamentos
        b = b0 if f is None else cv2.flip(b0, f)
        mad = float(np.abs(a - b).mean())
        if mad < best[0]:
            best = (mad, float(np.corrcoef(a.mean(2).ravel(), b.mean(2).ravel())[0, 1]))
    return best


def stage_analyze():
    from scipy.stats import chi2_contingency
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    E = {"gerado_em": time.strftime("%Y-%m-%d %H:%M"), "seed": SEED, "script": "scripts/m2_evidencias.py"}
    m = manifest()
    h = pd.read_csv(OUT / "hashes_v1.csv"); sc = pd.read_csv(OUT / "artefatos_scores_script.csv").drop(columns="task")
    m = m.merge(h, on="id", how="left").merge(sc, on="id", how="left")
    m = m.merge(labels(), left_on="id", right_on="image", how="left").drop(columns="image")
    grp = pd.read_csv(ROOT / DATA["task3_groupings"])
    m = m.merge(grp[["image", "lesion_id", "diagnosis_confirm_type"]], left_on="id", right_on="image", how="left").drop(columns="image")
    meta = pd.read_csv(ROOT / DATA["ham_metadata"])
    m = m.merge(meta[["image_id", "age", "sex", "localization"]], left_on="id", right_on="image_id", how="left").drop(columns="image_id")

    # 1. integridade e formato (medido)
    E["counts"] = {f"{t}/{s}": int(n) for (t, s), n in m.groupby(["task", "source"]).size().items()}
    E["preprocess_status"] = m.status.value_counts().to_dict()
    E["hash_status"] = m.hash_status.value_counts().to_dict()
    E["raw_mode"] = m.groupby("task").raw_mode.value_counts().unstack(fill_value=0).to_dict("index")
    E["raw_format"] = m.raw_format.value_counts().to_dict()
    cls_dims = m[m.task == "cls"].assign(d=lambda d: d.raw_w.astype(int).astype(str) + "x" + d.raw_h.astype(int).astype(str)).d.value_counts()
    E["cls_raw_dims"] = cls_dims.to_dict()
    s = m[m.task == "seg"]
    E["seg_raw_w_range"] = [int(s.raw_w.min()), int(s.raw_w.max())]; E["seg_raw_h_range"] = [int(s.raw_h.min()), int(s.raw_h.max())]
    E["seg_aspect_wh"] = {k: round(float(v), 3) for k, v in (s.raw_w / s.raw_h).describe()[["min", "50%", "max"]].items()}
    E["seg_mask_shape_matches_image"] = int((s.mask_raw_shape == s.raw_w.astype(int).astype(str) + "x" + s.raw_h.astype(int).astype(str)).sum())
    E["proc_shapes"] = m.groupby("task").apply(lambda d: sorted({f"{w}x{hh}x{c}" for w, hh, c in zip(d.proc_w, d.proc_h, d.proc_c)})).to_dict()
    E["mask_raw_binary"] = int(s.mask_raw_binary.sum()); E["mask_proc_binary"] = int(s.mask_proc_binary.sum())
    E["mask_area"] = {k: round(float(v), 4) for k, v in s.mask_area.describe(percentiles=[.05, .5, .95]).items() if k != "count"}
    E["mask_empty"] = int((s.mask_area == 0).sum()); E["mask_lt_1pct"] = int((s.mask_area < 0.01).sum())
    E["mask_gt_90pct"] = int((s.mask_area > 0.9).sum()); E["mask_touch_border"] = round(float(s.mask_touch_border.mean()), 4)
    E["mask_multi_component"] = int((s.mask_components > 1).sum())
    c = m[m.task == "cls"]
    E["cls_sem_rotulo"] = int(c.dx.isna().sum()); E["cls_multi_rotulo"] = int((c.n_pos != 1).sum())
    meta_chk = c[c.source == "train"].merge(meta[["image_id", "dx"]].rename(columns={"dx": "dx_meta"}), left_on="id", right_on="image_id")
    E["gt_vs_metadata_dx_divergente"] = int((meta_chk.dx.str.lower() != meta_chk.dx_meta).sum())

    # 2. duplicatas exatas (MD5 do bruto) e próximas (pHash do processado + checagem de pixels)
    E["md5_dup_groups"] = {}
    for name, sub in {"cls": m[m.task == "cls"], "seg": m[m.task == "seg"], "cross_task": m}.items():
        g = sub.groupby("md5_raw").id.count(); E["md5_dup_groups"][name] = int((g > 1).sum())
    ex = m[m.md5_raw.duplicated(keep=False)].sort_values("md5_raw")
    E["md5_dup_cross_split"] = int(ex.groupby("md5_raw").split.nunique().gt(1).sum()) if len(ex) else 0
    allp = []
    for task in ("cls", "seg"):
        sub = m[m.task == task].reset_index(drop=True)
        p = near_pairs(sub)
        for k in ("id", "split", "dx", "lesion_id", "path"):
            p[f"{k}_a"] = sub.loc[p.a, k].values; p[f"{k}_b"] = sub.loc[p.b, k].values
        p["task"] = task
        allp.append(p)
    pairs = pd.concat(allp, ignore_index=True)
    chk = [pixel_check(a, b) for a, b in zip(pairs.path_a, pairs.path_b)]
    pairs["mad"] = [x[0] for x in chk]; pairs["corr"] = [x[1] for x in chk]
    pairs["mesma_lesao"] = (pairs.lesion_id_a == pairs.lesion_id_b) & pairs.lesion_id_a.notna()
    pairs["entre_splits"] = pairs.split_a != pairs.split_b
    # critério de duplicata confirmada (derivado): diferença média < 5 níveis de cinza e correlação > 0,95
    pairs["confirmada"] = (pairs.mad < 5) & (pairs["corr"] > 0.95)
    pairs.drop(columns=["path_a", "path_b"]).to_csv(OUT / "duplicatas_phash_v1.csv", index=False)
    cp = pairs[pairs.task == "cls"]
    E["phash"] = dict(
        limiar_hamming=4,
        pares_cls=int(len(cp)), pares_seg=int((pairs.task == "seg").sum()),
        pares_cls_mesma_lesao=int(cp.mesma_lesao.sum()),
        pares_cls_entre_splits=int(cp.entre_splits.sum()),
        pares_cls_entre_splits_por_par={"|".join(sorted(k)): int(v) for k, v in cp[cp.entre_splits].groupby(["split_a", "split_b"]).size().items()},
        pares_rotulo_diferente=int((cp.dx_a != cp.dx_b).sum()),
        mad_mediana_mesma_lesao=round(float(cp[cp.mesma_lesao].mad.median()), 2),
        mad_mediana_lesoes_diferentes=round(float(cp[~cp.mesma_lesao].mad.median()), 2),
        mad_min_entre_splits=round(float(cp[cp.entre_splits].mad.min()), 2) if cp.entre_splits.any() else None,
        confirmadas_total=int(pairs.confirmada.sum()),
        confirmadas_entre_splits=int((pairs.confirmada & pairs.entre_splits).sum()),
        confirmadas_mesma_lesao=int((pairs.confirmada & pairs.mesma_lesao).sum()),
    )
    # grupos de duplicata confirmada (componentes conexos)
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    for a, b in zip(pairs[pairs.confirmada].id_a, pairs[pairs.confirmada].id_b):
        parent[find(a)] = find(b)
    ex_ids = ex.groupby("md5_raw").id.apply(list)
    for ids in ex_ids:
        for o in ids[1:]:
            parent[find(o)] = find(ids[0])
    roots = {i: find(i) for i in set(parent) | set(parent.values())}
    names = {r: f"DUP{k:03d}" for k, r in enumerate(sorted(set(roots.values())), 1)}
    m["duplicate_group"] = m.id.map(lambda i: names.get(roots.get(i)) if i in roots else "")
    E["duplicate_groups"] = int(len(names))
    dg = m[m.duplicate_group != ""].groupby("duplicate_group").split.nunique()
    E["duplicate_groups_cross_split"] = int((dg > 1).sum())

    # 3. artefatos: limiares contra rótulos manuais (n=200, conferidos pelo aluno)
    rot = pd.read_csv(OUT / "auditoria/rotulos_manuais.csv")
    r = rot.merge(m[["id", "task", "vignette", "hair", "ruler", "dark_border"]], on=["id", "task"])
    DET = {"vinheta": ("vignette", "menor"), "pelo": ("hair", "maior"), "regua": ("ruler", "maior")}
    LIM, val = {}, {}
    for art, (col, sent) in DET.items():
        y = r[art].astype(int).values; s_ = r[col].values if sent == "maior" else -r[col].values
        auc = roc_auc_score(y, s_)
        best = max(((precision_recall_fscore_support(y, (s_ >= t).astype(int), average="binary", zero_division=0)[:3], t) for t in np.unique(s_)), key=lambda z: z[0][2])
        (p_, rc, f1), t = best; thr = float(t if sent == "maior" else -t)
        LIM[col] = thr; val[art] = dict(auc=round(float(auc), 3), precisao=round(float(p_), 3), recall=round(float(rc), 3), f1=round(float(f1), 3), limiar=round(thr, 4), prevalencia_manual=round(float(y.mean()), 3), n=int(len(y)))
    # critério de uso (derivado): só detector com AUC ≥ 0,85 contra o rótulo humano entra em análises por subgrupo
    for art in val:
        val[art]["usar"] = bool(val[art]["auc"] >= 0.85)
    E["detectores_validados"] = val; E["limiares"] = LIM
    E["prevalencia_manual"] = {k: round(float(rot[k].mean()), 3) for k in ["pelo", "regua", "vinheta", "bolha", "caneta", "outro"]}
    E["prevalencia_manual_por_grupo"] = {k: {g: round(float(v), 3) for g, v in rot.groupby("dx")[k].mean().items()} for k in ["pelo", "regua", "vinheta"]}
    yaml.safe_dump({"version": "v1", "limiares": LIM, "validado_com_rotulos_manuais": True, "rotulos": "reports/m2_dados/auditoria/rotulos_manuais.csv (n=200; rotulados pelo assistente, conferidos integralmente por Valmir)", "metricas": val, "nota": "pelo e regua com AUC < 0,85: flags automáticas NÃO usadas; usar só a prevalência manual (n=200)"},
                   open(ROOT / "configs/artifacts.yaml", "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    m["tem_vinheta"] = m.vignette < LIM["vignette"]; m["tem_pelo"] = m.hair > LIM["hair"]; m["tem_regua"] = m.ruler > LIM["ruler"]
    flags = ["tem_vinheta", "tem_pelo", "tem_regua"]
    tr = m[(m.task == "cls") & (m.source == "train")]
    E["artefato_por_classe"] = tr.groupby("dx")[flags].mean().reindex(CLASSES).round(3).to_dict()
    E["artefato_por_split"] = m[m.task == "cls"].groupby("split")[flags].mean().round(3).to_dict()
    E["artefato_seg"] = m[m.task == "seg"][flags].mean().round(3).to_dict()
    for f in flags:
        chi2, pv, dof, _ = chi2_contingency(pd.crosstab(tr.dx, tr[f]))
        E[f"chi2_{f}"] = dict(chi2=round(float(chi2), 1), gl=int(dof), p=float(pv))

    # 4. distribuição por classe/split, metadados por split
    E["dist_cls"] = m[m.task == "cls"].groupby(["dx", "split"]).size().unstack(fill_value=0).reindex(CLASSES).to_dict()
    E["dist_seg"] = m[m.task == "seg"].split.value_counts().to_dict()
    tr_all = m[(m.task == "cls") & (m.source == "train")]
    E["meta_por_split"] = {
        sp: dict(n=int(len(d)), idade_media=round(float(d.age.mean()), 1), idade_faltante=int(d.age.isna().sum()),
                 sexo={k: round(float(v), 3) for k, v in d.sex.value_counts(normalize=True).items()},
                 histo=round(float((d.diagnosis_confirm_type == "histopathology").mean()), 3),
                 mel=round(float((d.dx == "MEL").mean()), 3))
        for sp, d in tr_all.groupby("split")}
    E["localizacao_top"] = tr_all.localization.value_counts().head(6).to_dict()

    # 5. pesos de classe e vazamento por lesão
    cnt = tr_all[tr_all.split == "train"].dx.value_counts().reindex(CLASSES)
    w = cnt.sum() / (len(CLASSES) * cnt); w = (w / w.mean()).round(3)
    json.dump({"version": "v1", "metodo": "inverso da frequencia, media 1", "contagem_treino": {k: int(v) for k, v in cnt.items()}, "pesos": w.to_dict()},
              open(ROOT / "data/splits/class_weights_v1.json", "w"), indent=2)
    E["class_weights"] = w.to_dict()
    E["lesoes_em_mais_de_um_split"] = int((tr_all.groupby("lesion_id").split.nunique() > 1).sum())
    E["ids_cls_seg_em_comum"] = int(len(set(m[m.task == "cls"].id) & set(m[m.task == "seg"].id)))
    sp = pd.read_csv(ROOT / DATA["split_file"]).set_index("image").split
    nonoff = m[m.source != "test_official"]
    E["split_file"] = DATA["split_file"]
    E["split_reconstruivel"] = bool((sp.reindex(nonoff.id).values == nonoff.split.values).all())
    seg_lid = pd.read_csv(ROOT / DATA["split_file"]).query("task == 'seg'").set_index("image").lesion_id
    E["seg_grupos_em_mais_de_um_split"] = int((m[m.task == "seg"].assign(g=lambda d: d.id.map(seg_lid)).groupby("g").split.nunique() > 1).sum())

    # 6. manifesto no formato do workbook
    m["label"] = np.where(m.task == "cls", m.dx, "lesao (mascara binaria)")
    m["group_id"] = np.where(m.lesion_id.notna(), m.lesion_id, np.where(m.seg_group.notna(), m.seg_group, m.id))
    m["capture_group"] = np.where(m.task == "cls", m.diagnosis_confirm_type.fillna("desconhecido (teste oficial)"), "task1")
    m["split_m2"] = m.split.map(SPLIT_NAME)
    issues = []
    for _, x in m.iterrows():
        it = []
        if x.task == "seg" and x.mask_touch_border: it.append("mascara_toca_borda")
        if x.task == "seg" and x.mask_area < 0.01: it.append("lesao_pequena_<1%")
        if x.tem_vinheta: it.append("vinheta")
        if x.duplicate_group: it.append("duplicata_mesmo_split")
        issues.append(";".join(it))
    m["issue_type"] = issues
    m["integrity_status"] = np.where((m.status.isin(["ok", "skip"])) & (m.hash_status == "ok"), "OK", "ERRO")
    out = pd.DataFrame(dict(
        sample_id=m.id, task=m.task, file_path=m.path, raw_path=m.raw_path, label=m.label, group_id=m.group_id,
        source=np.where(m.task == "cls", np.where(m.source == "train", "ISIC2018 Task3 (HAM10000) treino", "ISIC2018 Task3 teste oficial"), "ISIC2018 Task1 treino"),
        license=m.task.map(LICENSE), width=m.raw_w.astype(int), height=m.raw_h.astype(int), channels=3,
        annotation_path=np.where(m.task == "seg", m.mask_path, np.where(m.source == "test_official", DATA["task3_test_labels"], DATA["task3_labels"])),
        object_count=1, integrity_status=m.integrity_status, issue_type=m.issue_type, split=m.split_m2,
        file_hash=m.md5_raw, duplicate_group=m.duplicate_group, capture_group=m.capture_group,
        notes=np.where(m.task == "seg", "mask_area=" + m.mask_area.round(3).astype(str), "dx_type=" + m.diagnosis_confirm_type.fillna("NA")),
        include="Sim", exclusion_reason=""))
    out.to_csv(ROOT / "data/splits/manifest_m2_v1.csv", index=False)
    for s_ in ["train", "validation", "test", "holdout"]:
        out[out.split == s_].to_csv(ROOT / f"data/splits/{'test_official' if s_ == 'holdout' else s_}.csv", index=False)
    E["manifest_rows"] = {k: int(v) for k, v in out.split.value_counts().items()}
    E["integrity_erro"] = int((out.integrity_status != "OK").sum())
    m.to_pickle(OUT / "_m2_frame.pkl")
    json.dump(E, open(OUT / "evidencias_m2_script.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    print(json.dumps(E, indent=1, ensure_ascii=False, default=str))


def stage_figures():
    """Figuras 08–10 a partir do cache da análise. Augmentations com Albumentations lendo configs/transforms.yaml
    (mesma tradução do data.py, sem ToTensor/torch, para rodar em qualquer máquina)."""
    import albumentations as A
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    m = pd.read_pickle(OUT / "_m2_frame.pkl"); E = json.load(open(OUT / "evidencias_m2_script.json", encoding="utf-8"))
    # 08 — vinheta por classe (detector validado) + prevalência manual
    tr = m[(m.task == "cls") & (m.source == "train")]
    rate = tr.groupby("dx").tem_vinheta.mean().reindex(CLASSES) * 100
    man = pd.Series(E["prevalencia_manual_por_grupo"]["vinheta"]).reindex(CLASSES) * 100
    fig, ax = plt.subplots(figsize=(8, 3.4)); x = np.arange(len(CLASSES))
    ax.bar(x - .2, rate.values, .4, label="detector (todas as imagens)", color="#3b82f6"); ax.bar(x + .2, man.values, .4, label="rótulo manual (n=200)", color="#94a3b8")
    ax.set_xticks(x, CLASSES); ax.set_ylabel("% com vinheta"); ax.legend(frameon=False); ax.set_title("Vinheta por classe — Task 3 (treino+val+teste interno)")
    plt.tight_layout(); plt.savefig(FIG / "08b_vinheta_detector_vs_manual.png", dpi=120); plt.close()
    # 09 — augmentations: 1 imagem cls e 1 seg, 6 sorteios, máscara sobreposta
    BORDER = {"reflect": cv2.BORDER_REFLECT_101, "constant": cv2.BORDER_CONSTANT}
    def one(spec):
        spec = dict(spec); name = spec.pop("name")
        if "border_mode" in spec: spec["border_mode"] = BORDER[spec["border_mode"]]
        if name == "RandomResizedCrop": spec["size"] = tuple(spec["size"])
        if name == "ShiftScaleRotate":
            return A.Affine(translate_percent=spec["shift_limit"], scale=(1 - spec["scale_limit"], 1 + spec["scale_limit"]), rotate=(-spec["rotate_limit"], spec["rotate_limit"]), border_mode=spec["border_mode"], p=spec["p"])
        return getattr(A, name)(**spec)
    tc = A.Compose([one(s) for s in TR["classification"]["train"]], seed=SEED)
    ts = A.Compose([one(s) for s in TR["segmentation"]["train"]], seed=SEED)
    rng = np.random.default_rng(SEED)
    rc = tr[tr.dx == "MEL"].iloc[int(rng.integers((tr.dx == "MEL").sum()))]
    sg = m[(m.task == "seg") & (m.split == "train")]; rs = sg.iloc[int(rng.integers(len(sg)))]
    ic = cv2.cvtColor(cv2.imread(str(ROOT / rc.path)), cv2.COLOR_BGR2RGB)
    iseg = cv2.cvtColor(cv2.imread(str(ROOT / rs.path)), cv2.COLOR_BGR2RGB); mk = cv2.imread(str(ROOT / rs.mask_path), 0)
    fig, ax = plt.subplots(3, 7, figsize=(15, 6.6))
    ax[0, 0].imshow(ic); ax[0, 0].set_title(f"original {rc.id} ({rc.dx})", fontsize=7)
    ax[1, 0].imshow(iseg); ax[2, 0].imshow(iseg); ax[2, 0].contour(mk > 127, colors="lime", linewidths=.8); ax[1, 0].set_title(f"original {rs.id}", fontsize=7)
    for j in range(1, 7):
        ax[0, j].imshow(tc(image=ic)["image"]); ax[0, j].set_title("aug treino", fontsize=7)
        o = ts(image=iseg, mask=mk); ax[1, j].imshow(o["image"]); ax[2, j].imshow(o["image"]); ax[2, j].contour(o["mask"] > 127, colors="lime", linewidths=.8)
    for a in ax.ravel(): a.axis("off")
    plt.suptitle("linha 1: classificação · linhas 2–3: segmentação (contorno da máscara após a mesma transformação)", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "09b_augmentations_contorno.png", dpi=110); plt.close()
    # 10 — distribuição por classe e split
    d = pd.DataFrame(E["dist_cls"]).reindex(CLASSES)[["train", "val", "test", "test_official"]]
    pct = d / d.sum() * 100
    fig, ax = plt.subplots(figsize=(9, 3.6)); w = .2
    for k, (c, col) in enumerate(zip(pct.columns, ["#1e3a8a", "#3b82f6", "#93c5fd", "#f59e0b"])):
        ax.bar(np.arange(7) + (k - 1.5) * w, pct[c].values, w, label=f"{c} (n={int(d[c].sum())})", color=col)
    ax.set_xticks(np.arange(7), CLASSES); ax.set_ylabel("% do split"); ax.set_yscale("log"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Distribuição de classes por split (escala log) — split_v2 / teste oficial")
    plt.tight_layout(); plt.savefig(FIG / "10_distribuicao_split.png", dpi=120); plt.close()
    print("figuras 08, 09, 10 gravadas")


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("hash", "all"): stage_hash()
    if st in ("scores", "all"): stage_scores()
    if st in ("analyze", "all"): stage_analyze()
    if st in ("figures", "all"): stage_figures()
