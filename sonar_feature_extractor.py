"""
sonar_feature_extractor.py
==========================
Converte imagens JPEG de Side-Scan Sonar (SSS) + anotações YOLO (.txt)
em features numéricas salvas em CSV, prontas para modelos de ML.

Dataset de referência:
  Santos & Moura (2024) — "Side-scan sonar imaging for Mine detection"
  DOI: 10.6084/m9.figshare.24574879.v2

Formato do TXT (YOLO):
  <class_id> <x_center> <y_center> <width> <height>   (valores normalizados 0-1)
  class 0 = NOMBO (Non-Mine-like Bottom Object)
  class 1 = MILCO (Mine-Like Contact / objeto balístico)

── Uso rápido ───────────────────────────────────────────────────────────

  # Imagem única
  python sonar_feature_extractor.py --image 0257_2010.jpg

  # Uma pasta
  python sonar_feature_extractor.py --folder ./dataset/

  # Várias pastas explícitas
  python sonar_feature_extractor.py --folders ./train/ ./val/ ./test/

  # Pasta raiz + varredura recursiva de subpastas   
  python sonar_feature_extractor.py --folder ./dataset/ --recursive

  # Lista de pastas via arquivo de texto (uma por linha)
  python sonar_feature_extractor.py --folder-list pastas.txt

  # Combinação: tag de origem + sem parar em erros
  python sonar_feature_extractor.py --folders ./2010/ ./2021/ \\
      --tag-source --output merged.csv --mode per_object

──────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.filters import sobel
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONSTANTES DO DATASET
# ─────────────────────────────────────────────
CLASS_NAMES = {0: "NOMBO", 1: "MILCO"}   # conforme paper
GRID_SIZE   = 4     # divide imagem em grid 4x4 → 16 células espaciais
HIST_BINS   = 32    # bins do histograma de intensidade


# ═══════════════════════════════════════════════════════════════════
#  BLOCO 1 — LEITURA DE DADOS
# ═══════════════════════════════════════════════════════════════════

def load_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega imagem JPEG e retorna versão colorida (BGR) e grayscale.

    Por que OpenCV? 
      - Mais rápido que PIL para arrays numpy
      - Suporte nativo a conversões de espaço de cor
    """

    # AJUSTE FEITO AQUI PARA FUNCIONAR COM PASTAS QUE TEM ACENTO, COMO "Área de Trabalho"
    raw_bytes = Path(image_path).read_bytes()   # Python lida com Unicode nativamente
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # decodifica do buffer em memória

    if img_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_gray


def load_annotations(label_path: str, img_w: int, img_h: int) -> list[dict]:
    """
    Lê arquivo YOLO e converte coordenadas normalizadas → pixels absolutos.

    Retorna lista de dicts com campos:
      class_id, class_name, x_center_norm, y_center_norm, w_norm, h_norm,
      x1, y1, x2, y2 (pixel coords do bounding box)
    """
    annotations = []
    if not os.path.exists(label_path):
        return annotations   # sem anotações é válido (imagem negativa)

    with open(label_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), \
                                  float(parts[3]), float(parts[4])
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            annotations.append({
                "class_id":       cls,
                "class_name":     CLASS_NAMES.get(cls, f"class_{cls}"),
                "x_center_norm":  xc,
                "y_center_norm":  yc,
                "w_norm":         w,
                "h_norm":         h,
                "x1": max(0, x1), "y1": max(0, y1),
                "x2": min(img_w, x2), "y2": min(img_h, y2),
            })
    return annotations


# ═══════════════════════════════════════════════════════════════════
#  BLOCO 2 — EXTRAÇÃO DE FEATURES (imagem inteira)
# ═══════════════════════════════════════════════════════════════════

def features_basic_stats(gray: np.ndarray) -> dict:
    """
    Estatísticas de primeira ordem da intensidade dos pixels.
    Captura brilho médio, contraste global, assimetria da distribuição.
    """
    flat = gray.flatten().astype(np.float32)
    return {
        "mean":       float(np.mean(flat)),
        "std":        float(np.std(flat)),
        "min":        float(np.min(flat)),
        "max":        float(np.max(flat)),
        "p10":        float(np.percentile(flat, 10)),
        "p25":        float(np.percentile(flat, 25)),
        "p50":        float(np.percentile(flat, 50)),
        "p75":        float(np.percentile(flat, 75)),
        "p90":        float(np.percentile(flat, 90)),
        "skewness":   float(skew(flat)),
        "kurtosis":   float(kurtosis(flat)),
        "iqr":        float(np.percentile(flat, 75) - np.percentile(flat, 25)),
        "energy":     float(np.sum(flat ** 2) / flat.size),   # energia normalizada
    }


def features_histogram(gray: np.ndarray, bins: int = HIST_BINS) -> dict:
    """
    Histograma de intensidade normalizado.
    Captura a distribuição de tons — muito útil para distinguir
    regiões de sombra acústica (valores baixos) de reflexos (altos).
    """
    hist, _ = np.histogram(gray.flatten(), bins=bins, range=(0, 256), density=True)
    return {f"hist_bin_{i:02d}": float(v) for i, v in enumerate(hist)}


def features_texture_glcm(gray: np.ndarray) -> dict:
    """
    Gray Level Co-occurrence Matrix (GLCM) — Haralick features.

    Captura padrões de textura como rugosidade do fundo marinho,
    homogeneidade e repetição de estruturas, essencial para distinguir
    objetos artificiais (minas) do fundo natural.

    Ângulos: 0°, 45°, 90°, 135° → média e desvio por propriedade.
    """
    # Reduz para 64 níveis para economizar memória e ruído
    gray_64 = (gray // 4).astype(np.uint8)
    distances = [1, 3]
    angles    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm  = graycomatrix(gray_64, distances=distances, angles=angles,
                          levels=64, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy",
             "correlation", "ASM"]
    feats = {}
    for prop in props:
        vals = graycoprops(glcm, prop).flatten()
        feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
        feats[f"glcm_{prop}_std"]  = float(np.std(vals))
    return feats


def features_gradient(gray: np.ndarray) -> dict:
    """
    Features baseadas em gradiente (bordas e transições).

    No sonar SSS, objetos sólidos criam bordas nítidas entre a
    "highlight" (reflexo) e a "shadow" (sombra acústica).
    """
    sobel_mag  = sobel(gray.astype(np.float32))
    laplacian  = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "sobel_mean":     float(np.mean(sobel_mag)),
        "sobel_std":      float(np.std(sobel_mag)),
        "sobel_max":      float(np.max(sobel_mag)),
        "laplacian_mean": float(np.mean(np.abs(laplacian))),
        "laplacian_std":  float(np.std(laplacian)),
        "laplacian_var":  float(np.var(laplacian)),   # proxy de nitidez
    }


def features_frequency(gray: np.ndarray) -> dict:
    """
    Energia no domínio da frequência (FFT).

    Objetos balísticos introduzem padrões periódicos de alta frequência.
    Dividimos o espectro em bandas: baixa, média e alta frequência.
    """
    fft   = np.fft.fft2(gray.astype(np.float32))
    fft_s = np.fft.fftshift(fft)
    mag   = np.abs(fft_s)

    h, w  = mag.shape
    cy, cx = h // 2, w // 2
    # Raios que delimitam bandas (em pixels no espaço de frequência)
    r_low, r_mid = min(h, w) // 8, min(h, w) // 4

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    low_energy  = float(np.sum(mag[dist <= r_low] ** 2))
    mid_energy  = float(np.sum(mag[(dist > r_low) & (dist <= r_mid)] ** 2))
    high_energy = float(np.sum(mag[dist > r_mid] ** 2))
    total       = low_energy + mid_energy + high_energy + 1e-9

    return {
        "fft_low_energy_ratio":  low_energy  / total,
        "fft_mid_energy_ratio":  mid_energy  / total,
        "fft_high_energy_ratio": high_energy / total,
        "fft_total_log_energy":  float(np.log1p(total)),
    }


def features_spatial_grid(gray: np.ndarray, grid: int = GRID_SIZE) -> dict:
    """
    Divide a imagem em um grid NxN e calcula média/desvio por célula.

    Captura assimetrias espaciais — ex: objetos no flanco direito do sonar
    geram padrões distintos de sombra na metade direita da imagem.
    """
    h, w   = gray.shape
    ch, cw = h // grid, w // grid
    feats  = {}
    for r in range(grid):
        for c in range(grid):
            cell = gray[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].astype(np.float32)
            feats[f"grid_r{r}c{c}_mean"] = float(np.mean(cell))
            feats[f"grid_r{r}c{c}_std"]  = float(np.std(cell))
    return feats


def features_hog_condensed(gray: np.ndarray) -> dict:
    """
    Histogram of Oriented Gradients (HOG) — versão condensada.

    HOG completo pode ter milhares de dimensões.
    Aqui usamos cells grandes para obter um vetor compacto (~36 valores)
    que captura a estrutura direcional da textura sonar.
    """
    resized = cv2.resize(gray, (128, 128))
    hog_vec = hog(resized, orientations=9, pixels_per_cell=(32, 32),
                  cells_per_block=(2, 2), feature_vector=True)
    return {f"hog_{i:03d}": float(v) for i, v in enumerate(hog_vec)}


def features_color_channels(img_bgr: np.ndarray) -> dict:
    """
    Estatísticas por canal de cor (B, G, R).

    Imagens SSS frequentemente são pseudocoloradas (como neste dataset —
    tons âmbar). O canal G pode capturar realces específicos do sensor.
    """
    feats = {}
    for i, ch_name in enumerate(["B", "G", "R"]):
        ch = img_bgr[:, :, i].astype(np.float32)
        feats[f"ch_{ch_name}_mean"] = float(np.mean(ch))
        feats[f"ch_{ch_name}_std"]  = float(np.std(ch))
        feats[f"ch_{ch_name}_p50"]  = float(np.percentile(ch, 50))
    return feats


# ═══════════════════════════════════════════════════════════════════
#  BLOCO 3 — FEATURES DA REGIÃO DO OBJETO (bounding box)
# ═══════════════════════════════════════════════════════════════════

def features_object_region(gray: np.ndarray, ann: dict) -> dict:
    """
    Extrai features dentro e ao redor do bounding box de cada objeto.

    Inclui:
      - Estatísticas da ROI (Region of Interest)
      - Razão highlight/shadow dentro da região (proxy acústico)
      - Contraste local: ROI vs. vizinhança imediata
    """
    x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
    roi = gray[y1:y2, x1:x2].astype(np.float32)

    if roi.size == 0:
        return {}

    roi_mean = float(np.mean(roi))
    roi_std  = float(np.std(roi))

    # Região ao redor (contexto local) — expande o bbox em 50%
    pad = max(int((x2 - x1) * 0.5), int((y2 - y1) * 0.5), 10)
    cx1 = max(0, x1 - pad);  cy1 = max(0, y1 - pad)
    cx2 = min(gray.shape[1], x2 + pad)
    cy2 = min(gray.shape[0], y2 + pad)
    ctx = gray[cy1:cy2, cx1:cx2].astype(np.float32)
    ctx_mean = float(np.mean(ctx))

    # Limiar de Otsu para separar highlight de shadow dentro da ROI
    _, thresh = cv2.threshold(roi.astype(np.uint8), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    highlight_ratio = float(np.sum(thresh > 0) / thresh.size) if thresh.size > 0 else 0.0

    return {
        "obj_roi_mean":         roi_mean,
        "obj_roi_std":          roi_std,
        "obj_roi_min":          float(np.min(roi)),
        "obj_roi_max":          float(np.max(roi)),
        "obj_roi_skewness":     float(skew(roi.flatten())),
        "obj_roi_energy":       float(np.sum(roi ** 2) / roi.size),
        "obj_highlight_ratio":  highlight_ratio,
        "obj_local_contrast":   abs(roi_mean - ctx_mean),   # contraste com contexto
        "obj_area_pixels":      float((x2 - x1) * (y2 - y1)),
        "obj_aspect_ratio":     float((x2 - x1) / max(y2 - y1, 1)),
    }


# ═══════════════════════════════════════════════════════════════════
#  BLOCO 4 — ORQUESTRADOR: image → linha(s) do CSV
# ═══════════════════════════════════════════════════════════════════

def extract_features(
    image_path: str,
    label_path: str | None = None,
    mode: str = "per_image"
) -> list[dict]:
    """
    Extrai todas as features de uma imagem e suas anotações.

    Parâmetros
    ----------
    image_path : str
        Caminho para a imagem JPEG.
    label_path : str | None
        Caminho para o .txt YOLO. Se None, tenta inferir pelo mesmo stem.
    mode : str
        "per_image"  → 1 linha por imagem (para classificação global)
        "per_object" → 1 linha por objeto anotado (para detecção por objeto)

    Retorna
    -------
    list[dict] : linhas a serem inseridas no DataFrame/CSV
    """
    # ── Caminhos ────────────────────────────────────────────────────
    image_path = str(image_path)
    if label_path is None:
        stem = Path(image_path).stem
        label_path = str(Path(image_path).parent / f"{stem}.txt")

    # ── Carregamento ────────────────────────────────────────────────
    img_bgr, gray = load_image(image_path)
    h, w = gray.shape
    annotations = load_annotations(label_path, w, h)

    # ── Features globais (toda a imagem) ────────────────────────────
    global_feats = {
        "image_path":    image_path,
        "img_width":     w,
        "img_height":    h,
        "n_annotations": len(annotations),
        "has_milco":     int(any(a["class_id"] == 1 for a in annotations)),
        "has_nombo":     int(any(a["class_id"] == 0 for a in annotations)),
    }
    global_feats.update(features_basic_stats(gray))
    global_feats.update(features_histogram(gray))
    global_feats.update(features_texture_glcm(gray))
    global_feats.update(features_gradient(gray))
    global_feats.update(features_frequency(gray))
    global_feats.update(features_spatial_grid(gray))
    global_feats.update(features_hog_condensed(gray))
    global_feats.update(features_color_channels(img_bgr))

    # ── Montagem das linhas ──────────────────────────────────────────
    if mode == "per_image":
        # Uma linha por imagem; agrega estatísticas das anotações
        if annotations:
            global_feats["ann_xc_mean"]   = float(np.mean([a["x_center_norm"] for a in annotations]))
            global_feats["ann_yc_mean"]   = float(np.mean([a["y_center_norm"] for a in annotations]))
            global_feats["ann_w_mean"]    = float(np.mean([a["w_norm"]        for a in annotations]))
            global_feats["ann_h_mean"]    = float(np.mean([a["h_norm"]        for a in annotations]))
            global_feats["ann_area_mean"] = float(np.mean([a["w_norm"] * a["h_norm"] for a in annotations]))
        else:
            for key in ["ann_xc_mean", "ann_yc_mean", "ann_w_mean",
                        "ann_h_mean", "ann_area_mean"]:
                global_feats[key] = np.nan
        # Label de classificação (1 = MILCO presente, 0 = sem MILCO)
        global_feats["label"] = global_feats["has_milco"]
        return [global_feats]

    elif mode == "per_object":
        if not annotations:
            # Imagem negativa: linha única sem objeto
            global_feats.update({
                "ann_class_id": -1, "ann_class_name": "negative",
                "ann_x_center_norm": np.nan, "ann_y_center_norm": np.nan,
                "ann_w_norm": np.nan, "ann_h_norm": np.nan,
                "label": 0,
            })
            return [global_feats]

        rows = []
        for ann in annotations:
            row = dict(global_feats)   # copia features globais
            row.update({
                "ann_class_id":        ann["class_id"],
                "ann_class_name":      ann["class_name"],
                "ann_x_center_norm":   ann["x_center_norm"],
                "ann_y_center_norm":   ann["y_center_norm"],
                "ann_w_norm":          ann["w_norm"],
                "ann_h_norm":          ann["h_norm"],
                "label":               ann["class_id"],   # 0=NOMBO, 1=MILCO
            })
            row.update(features_object_region(gray, ann))
            rows.append(row)
        return rows

    else:
        raise ValueError(f"mode deve ser 'per_image' ou 'per_object'. Recebido: {mode}")


# ═══════════════════════════════════════════════════════════════════
#  BLOCO 5 — PIPELINE: pasta(s) → CSV
# ═══════════════════════════════════════════════════════════════════

def _collect_images(folder: Path, recursive: bool) -> list[Path]:
    """
    Coleta todos os arquivos JPEG dentro de uma pasta.

    Parâmetros
    ----------
    folder    : diretório a varrer
    recursive : se True, desce em todos os subdiretórios
    """
    patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
    images: list[Path] = []
    for pattern in patterns:
        glob_fn = folder.rglob if recursive else folder.glob
        images.extend(glob_fn(pattern))
    # Remove duplicatas que podem surgir de padrões sobrepostos e ordena
    return sorted(set(images))


def _resolve_folders(
    folder: str | None,
    folders: list[str] | None,
    folder_list: str | None,
    recursive: bool,
) -> list[Path]:
    """
    Unifica as três formas de informar pastas em uma lista única e validada.

    Fontes aceitas (podem ser combinadas):
      1. --folder   : pasta única
      2. --folders  : N pastas explícitas na linha de comando
      3. --folder-list: arquivo de texto com uma pasta por linha
      4. --recursive : se ativo, desce em subpastas de qualquer fonte acima

    Retorna
    -------
    Lista de Path de TODAS as pastas a processar, sem duplicatas.
    """
    raw: list[str] = []

    if folder:
        raw.append(folder)

    if folders:
        raw.extend(folders)

    if folder_list:
        list_path = Path(folder_list)
        if not list_path.exists():
            raise FileNotFoundError(f"Arquivo de lista não encontrado: {folder_list}")
        with open(list_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):   # ignora comentários
                    raw.append(stripped)

    if not raw:
        raise ValueError(
            "Nenhuma pasta informada. Use --folder, --folders ou --folder-list."
        )

    resolved: list[Path] = []
    seen: set[Path] = set()

    for p in raw:
        path = Path(p).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Pasta não encontrada: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Não é um diretório: {path}")

        if recursive:
            # Expande para incluir todos os subdiretórios que contêm imagens
            subdirs = {img.parent for img in path.rglob("*.jpg")}
            subdirs |= {img.parent for img in path.rglob("*.jpeg")}
            subdirs.add(path)   # inclui a raiz também
            for sub in sorted(subdirs):
                if sub not in seen:
                    resolved.append(sub)
                    seen.add(sub)
        else:
            if path not in seen:
                resolved.append(path)
                seen.add(path)

    return resolved


def process_folder(
    folder: str,
    output_csv: str = "sonar_features.csv",
    mode: str = "per_object",
    tag_source: bool = False,
    skip_errors: bool = True,
    verbose: bool = True,
    recursive: bool = False,
) -> pd.DataFrame:
    """
    Processa todos os JPEGs de UMA pasta e retorna DataFrame.

    Parâmetros
    ----------
    folder      : pasta raiz com imagens + labels (.jpg + .txt lado a lado)
    output_csv  : caminho de saída do CSV (usado apenas quando chamada diretamente)
    mode        : "per_image" ou "per_object"
    tag_source  : adiciona coluna 'source_folder' com o nome da pasta de origem
    skip_errors : se True, imagens com erro são puladas (log de aviso); se False,
                  qualquer erro aborta o processo
    verbose     : imprime progresso linha a linha
    recursive   : varre subpastas (usado ao chamar diretamente, não via process_folders)
    """
    folder_path = Path(folder).resolve()
    images = _collect_images(folder_path, recursive=recursive)

    if not images:
        msg = f"Nenhuma imagem .jpg encontrada em: {folder_path}"
        if skip_errors:
            if verbose:
                print(f"  ⚠  {msg}")
            return pd.DataFrame()
        raise FileNotFoundError(msg)

    all_rows: list[dict] = []
    errors: list[str]   = []

    for i, img_path in enumerate(images, 1):
        lbl_path = img_path.with_suffix(".txt")
        try:
            rows = extract_features(str(img_path), str(lbl_path), mode=mode)
            if tag_source:
                for row in rows:
                    row["source_folder"] = folder_path.name
            all_rows.extend(rows)
            if verbose:
                print(f"  [{i:>4}/{len(images)}] ✓  {img_path.name}"
                      f"  →  {len(rows)} linha(s)")
        except Exception as exc:
            msg = f"{img_path.name} → {exc}"
            errors.append(msg)
            if verbose:
                print(f"  [{i:>4}/{len(images)}] ✗  {msg}")
            if not skip_errors:
                raise

    if errors and verbose:
        print(f"\n  ⚠  {len(errors)} erro(s) nesta pasta.")

    return pd.DataFrame(all_rows)


def process_folders(
    folders: list[str],
    output_csv: str = "sonar_features.csv",
    mode: str = "per_object",
    tag_source: bool = True,
    skip_errors: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Processa MÚLTIPLAS pastas e consolida tudo em um único CSV.

    Esta é a função principal para pipelines com datasets distribuídos
    em vários diretórios (ex: dados de anos diferentes, missões diferentes,
    splits train/val/test).

    Parâmetros
    ----------
    folders     : lista de caminhos de pastas a processar
    output_csv  : caminho do CSV consolidado de saída
    mode        : "per_image" ou "per_object"
    tag_source  : adiciona coluna 'source_folder' identificando a pasta de origem
                  de cada linha — fundamental para rastreabilidade e para criar
                  splits de treino/teste baseados em missão/ano
    skip_errors : pula pastas ou imagens com erro sem abortar o pipeline
    verbose     : imprime progresso detalhado

    Retorna
    -------
    pd.DataFrame consolidado com todas as linhas de todas as pastas

    Exemplo de uso programático
    ---------------------------
    >>> df = process_folders(
    ...     folders=["./data/2010/", "./data/2015/", "./data/2021/"],
    ...     output_csv="all_features.csv",
    ...     tag_source=True,
    ...     mode="per_object",
    ... )
    """
    all_dfs:      list[pd.DataFrame] = []
    folder_stats: list[dict]         = []

    separator = "─" * 60

    print(f"\n{'═' * 60}")
    print(f"  PROCESSAMENTO MULTI-PASTA")
    print(f"  Pastas encontradas : {len(folders)}")
    print(f"  Modo               : {mode}")
    print(f"  Tag de origem      : {'sim' if tag_source else 'não'}")
    print(f"  Pular erros        : {'sim' if skip_errors else 'não'}")
    print(f"  Saída              : {output_csv}")
    print(f"{'═' * 60}\n")

    for idx, folder in enumerate(folders, 1):
        folder_path = Path(folder).resolve()
        print(f"[{idx:>2}/{len(folders)}] 📂  {folder_path}")
        print(separator)

        df_folder = process_folder(
            folder=str(folder_path),
            mode=mode,
            tag_source=tag_source,
            skip_errors=skip_errors,
            verbose=verbose,
        )

        n_rows   = len(df_folder)
        n_images = len(_collect_images(folder_path, recursive=False))

        folder_stats.append({
            "pasta":    str(folder_path),
            "nome":     folder_path.name,
            "imagens":  n_images,
            "linhas":   n_rows,
            "status":   "ok" if n_rows > 0 else "vazio",
        })

        if not df_folder.empty:
            all_dfs.append(df_folder)

        print(f"  → {n_rows} linhas extraídas de {n_images} imagens\n")

    # ── Consolidação ────────────────────────────────────────────────
    if not all_dfs:
        raise RuntimeError(
            "Nenhuma linha foi extraída de nenhuma pasta. "
            "Verifique os caminhos e o formato das imagens."
        )

    df_final = pd.concat(all_dfs, ignore_index=True)

    # ── Reordenação de colunas: metadados primeiro, label por último ─
    priority_first = ["image_path", "source_folder", "img_width", "img_height",
                      "n_annotations", "has_milco", "has_nombo"]
    priority_last  = ["label"]

    first_cols = [c for c in priority_first if c in df_final.columns]
    last_cols  = [c for c in priority_last  if c in df_final.columns]
    mid_cols   = [c for c in df_final.columns
                  if c not in set(first_cols) | set(last_cols)]

    df_final = df_final[first_cols + mid_cols + last_cols]

    # ── Salva ────────────────────────────────────────────────────────
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    # ── Relatório final ──────────────────────────────────────────────
    stats_df = pd.DataFrame(folder_stats)

    print(f"\n{'═' * 60}")
    print("  RELATÓRIO FINAL")
    print(f"{'═' * 60}")
    print(f"\n  Por pasta:\n")
    for _, row in stats_df.iterrows():
        status_icon = "✅" if row["status"] == "ok" else "⚠️ "
        print(f"  {status_icon}  {row['nome']:30s}  "
              f"{row['imagens']:>5} imgs  →  {row['linhas']:>6} linhas")

    print(f"\n  {'─' * 50}")
    print(f"  Total de pastas    : {len(folders)}")
    print(f"  Pastas com dados   : {(stats_df['status'] == 'ok').sum()}")
    print(f"  Total de linhas    : {len(df_final)}")
    print(f"  Total de colunas   : {len(df_final.columns)}")

    if tag_source and "source_folder" in df_final.columns:
        print(f"\n  Linhas por pasta (source_folder):")
        for src, cnt in df_final["source_folder"].value_counts().items():
            print(f"    {src:30s}  {cnt:>6} linhas")

    if "label" in df_final.columns:
        print(f"\n  Distribuição de labels:")
        for lbl, cnt in df_final["label"].value_counts().sort_index().items():
            name = {0: "NOMBO", 1: "MILCO"}.get(lbl, str(lbl))
            print(f"    label={lbl} ({name:5s})  {cnt:>6} amostras  "
                  f"({cnt / len(df_final) * 100:.1f}%)")

    print(f"\n  ✅  CSV salvo em: {output_path.resolve()}")
    print(f"{'═' * 60}\n")

    return df_final


# ═══════════════════════════════════════════════════════════════════
#  BLOCO 6 — ANÁLISE RÁPIDA DO CSV GERADO
# ═══════════════════════════════════════════════════════════════════

def summarize_csv(df: pd.DataFrame) -> None:
    """Imprime um resumo rápido do DataFrame gerado."""
    print("\n" + "=" * 60)
    print("RESUMO DO DATASET GERADO")
    print("=" * 60)
    print(f"Total de amostras : {len(df)}")
    if "label" in df.columns:
        print(f"Distribuição de labels:\n{df['label'].value_counts().to_string()}")
    if "ann_class_name" in df.columns:
        print(f"\nDistribuição de classes:\n{df['ann_class_name'].value_counts().to_string()}")
    numeric = df.select_dtypes(include=np.number)
    print(f"\nFeatures numéricas : {numeric.shape[1]}")
    missing = numeric.isnull().sum()
    if missing.any():
        print(f"Colunas com NaN   : {(missing > 0).sum()}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
#  BLOCO 7 — CLI (linha de comando)
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extrai features numéricas de imagens SSS para ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Imagem única
  python sonar_feature_extractor.py --image foto.jpg

  # Uma pasta
  python sonar_feature_extractor.py --folder ./dataset/

  # Uma pasta + subpastas (recursivo)
  python sonar_feature_extractor.py --folder ./dataset/ --recursive

  # Várias pastas explícitas
  python sonar_feature_extractor.py --folders ./train/ ./val/ ./test/

  # Lista de pastas em arquivo de texto
  python sonar_feature_extractor.py --folder-list pastas.txt

  # Combinação com todas as opções
  python sonar_feature_extractor.py \\
      --folders ./2010/ ./2015/ ./2021/ \\
      --output merged.csv \\
      --mode per_object \\
      --tag-source \\
      --skip-errors
        """,
    )

    # ── Fontes de entrada (pelo menos uma obrigatória) ──────────────
    input_group = parser.add_argument_group("Fontes de entrada")
    input_group.add_argument(
        "--image", type=str, metavar="ARQUIVO",
        help="Caminho de uma única imagem JPEG."
    )
    input_group.add_argument(
        "--folder", type=str, metavar="DIR",
        help="Uma pasta com imagens + labels (.jpg e .txt lado a lado)."
    )
    input_group.add_argument(
        "--folders", type=str, nargs="+", metavar="DIR",
        help="Duas ou mais pastas separadas por espaço.\n"
             "Ex: --folders ./train/ ./val/ ./test/"
    )
    input_group.add_argument(
        "--folder-list", type=str, metavar="ARQUIVO",
        help="Arquivo .txt com uma pasta por linha (linhas com # são ignoradas).\n"
             "Pode ser combinado com --folder e --folders."
    )

    # ── Opções de varredura ─────────────────────────────────────────
    scan_group = parser.add_argument_group("Opções de varredura")
    scan_group.add_argument(
        "--recursive", action="store_true",
        help="Desce recursivamente em todas as subpastas de cada diretório informado."
    )

    # ── Opções de saída ─────────────────────────────────────────────
    out_group = parser.add_argument_group("Saída")
    out_group.add_argument(
        "--output", type=str, default="sonar_features.csv", metavar="CSV",
        help="Caminho do CSV gerado (default: sonar_features.csv).\n"
             "Diretórios intermediários são criados automaticamente."
    )
    out_group.add_argument(
        "--mode", type=str, default="per_object",
        choices=["per_image", "per_object"],
        help="per_object → 1 linha por objeto anotado (default).\n"
             "per_image  → 1 linha por imagem."
    )
    out_group.add_argument(
        "--tag-source", action="store_true",
        help="Adiciona coluna 'source_folder' com o nome da pasta de origem "
             "de cada linha. Recomendado ao combinar múltiplas pastas."
    )

    # ── Comportamento de erros ──────────────────────────────────────
    err_group = parser.add_argument_group("Tratamento de erros")
    err_group.add_argument(
        "--skip-errors", action="store_true", default=True,
        help="Pula imagens com erro e continua (comportamento padrão)."
    )
    err_group.add_argument(
        "--fail-fast", action="store_true",
        help="Aborta imediatamente no primeiro erro (substitui --skip-errors)."
    )

    # ── Outros ─────────────────────────────────────────────────────
    parser.add_argument(
        "--label", type=str, default=None, metavar="TXT",
        help="Caminho do .txt YOLO (apenas com --image). "
             "Inferido automaticamente se omitido."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suprime o log linha a linha (mantém apenas o relatório final)."
    )

    args = parser.parse_args()

    # ── Validação ───────────────────────────────────────────────────
    has_input = any([args.image, args.folder, args.folders, args.folder_list])
    if not has_input:
        parser.error(
            "Informe ao menos uma fonte de entrada: "
            "--image, --folder, --folders ou --folder-list"
        )

    skip_errors = not args.fail_fast
    verbose     = not args.quiet

    # ── Modo imagem única ───────────────────────────────────────────
    if args.image:
        rows = extract_features(args.image, args.label, mode=args.mode)
        df   = pd.DataFrame(rows)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"✅ {len(rows)} linha(s) salva(s) em: {args.output}")
        summarize_csv(df)
        return

    # ── Resolve todas as pastas informadas ──────────────────────────
    resolved = _resolve_folders(
        folder      = args.folder,
        folders     = args.folders,
        folder_list = args.folder_list,
        recursive   = args.recursive,
    )

    if len(resolved) == 1:
        # Caminho de pasta única — usa process_folder diretamente
        df = process_folder(
            folder      = str(resolved[0]),
            output_csv  = args.output,
            mode        = args.mode,
            tag_source  = args.tag_source,
            skip_errors = skip_errors,
            verbose     = verbose,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if "label" in df.columns:
            df = df[[c for c in df.columns if c != "label"] + ["label"]]
        df.to_csv(args.output, index=False)
        print(f"\n✅ CSV salvo em: {Path(args.output).resolve()}")
        summarize_csv(df)
    else:
        # Múltiplas pastas — usa process_folders
        df = process_folders(
            folders     = [str(f) for f in resolved],
            output_csv  = args.output,
            mode        = args.mode,
            tag_source  = args.tag_source,
            skip_errors = skip_errors,
            verbose     = verbose,
        )
        summarize_csv(df)


if __name__ == "__main__":
    main()
