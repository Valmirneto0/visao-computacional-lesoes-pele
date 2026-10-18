"""Detectores simples de artefatos de aquisição (M2). São *proxies* heurísticos, não classificadores:
servem para estratificar a análise de erro (com/sem artefato) e para levantar hipóteses.
A precisão de cada um é medida contra rótulos manuais em notebooks/02_auditoria.ipynb.

Todos recebem imagem BGR uint8 (como cv2.imread) e devolvem um score contínuo; o limiar é decidido
na auditoria e gravado em configs/artifacts.yaml.
"""
import cv2
import numpy as np


def vignette_score(img, frac=0.05):
    """Brilho médio (0-255) dos 4 cantos. Baixo = vinheta (círculo escuro do dermatoscópio)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); h, w = g.shape; k = max(2, int(min(h, w) * frac))
    return float(np.mean([g[:k, :k].mean(), g[:k, -k:].mean(), g[-k:, :k].mean(), g[-k:, -k:].mean()]))


def dark_border_fraction(img, thr=40):
    """Fração de pixels escuros (< thr) numa moldura de 4% da imagem. Complementa vignette_score:
    pega vinhetas assimétricas / recortes circulares descentrados."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); h, w = g.shape; b = max(2, int(0.04 * min(h, w)))
    border = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
    return float((border < thr).mean())


def hair_score(img, kernel=15, thr=20, min_len=25):
    """Fração de pixels em estruturas 'de pelo': black-hat morfológico (traços finos e escuros sobre
    fundo claro — DullRazor simplificado) → limiar → mantém só componentes compridos e finos
    (comprimento ≥ min_len px e área ≤ 60% da caixa envolvente). Calculado em 341x256 para ser
    independente da resolução. Confunde-se com bordas nítidas de lesão e perde pelos claros."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.shape != (256, 341):
        g = cv2.resize(g, (341, 256), interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    m = (cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k) > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(m.shape, bool)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        L = max(w, h)
        if L >= min_len and a / (L * max(1, min(w, h))) < 0.6:
            keep |= lab == i
    return float(keep.mean())


def ruler_score(img):
    """Proxy de régua: fração de segmentos de reta curtos, paralelos e regularmente espaçados
    (marcas da escala). Usa HoughLinesP no canal de bordas; devolve nº de segmentos por megapixel
    com ângulo dominante. É o detector mais fraco dos quatro — confirmar com rótulo manual."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 80, 160)
    lines = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=25, minLineLength=8, maxLineGap=2)
    if lines is None:
        return 0.0
    lines = np.asarray(lines).reshape(-1, 4)  # OpenCV 4 devolve (N,1,4); OpenCV 5 devolve (N,4)
    ang = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])) % 180
    hist, _ = np.histogram(ang, bins=18, range=(0, 180))
    dominant = hist.max()  # muitos segmentos no mesmo ângulo = marcas de régua
    return float(dominant / (g.size / 1e6))


def all_scores(img):
    return dict(vignette=vignette_score(img), dark_border=dark_border_fraction(img), hair=hair_score(img), ruler=ruler_score(img))
