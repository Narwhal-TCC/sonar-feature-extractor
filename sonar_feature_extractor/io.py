"""
io.py
=====
Leitura de dados do disco → objetos tipados.

Suporta dois modos de anotação:
  • Annotation (bounding box YOLO) — para SSS com objetos localizados
  • ImageLevelAnnotation            — para FLS onde a classe é da imagem inteira

SonarSample carrega ambos. Os extractors recebem sempre um SonarSample
e consultam sample.annotations (bounding boxes) ou sample.image_labels
(classificação de imagem inteira), conforme o dataset.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import CLASS_NAMES


# ── Tipos de anotação ─────────────────────────────────────────────────────────

@dataclass
class Annotation:
    """
    Bounding box YOLO + metadados.
    Usado por SSS (Side-Scan Sonar) onde objetos têm localização 2D.
    """
    class_id:      int
    class_name:    str
    x_center_norm: float
    y_center_norm: float
    w_norm:        float
    h_norm:        float
    x1: int;  y1: int
    x2: int;  y2: int

    @property
    def width(self)        -> int:   return self.x2 - self.x1
    @property
    def height(self)       -> int:   return self.y2 - self.y1
    @property
    def area_pixels(self)  -> int:   return self.width * self.height
    @property
    def aspect_ratio(self) -> float: return self.width / max(self.height, 1)


@dataclass
class ImageLevelAnnotation:
    """
    Classificação no nível da imagem inteira (sem bounding box).
    Usado por datasets FLS onde a classe é do objeto imageado, não de uma ROI.

    O campo `metadata` carrega informações adicionais específicas do dataset,
    como altura do sonar, dimensões do objeto, pose 6-DOF, etc.
    """
    class_id:   int
    class_name: str
    metadata:   Dict[str, Any] = field(default_factory=dict)


# ── Objeto central de dados ───────────────────────────────────────────────────

@dataclass
class SonarSample:
    """
    Imagem sonar carregada em memória + todas as suas anotações.

    Campos:
      image_path   : caminho original do arquivo
      bgr          : imagem colorida (H, W, 3) uint8
      gray         : imagem grayscale (H, W) uint8
      annotations  : bounding boxes YOLO (para SSS)
      image_labels : classificações de imagem inteira (para FLS)

    Os dois campos de anotação são independentes. Um dataset SSS popula
    `annotations`; um dataset FLS popula `image_labels`. Ambos podem
    coexistir em datasets híbridos futuros.
    """
    image_path:   Path
    bgr:          np.ndarray
    gray:         np.ndarray
    annotations:  List[Annotation]           = field(default_factory=list)
    image_labels: List[ImageLevelAnnotation] = field(default_factory=list)

    @property
    def width(self)  -> int: return int(self.gray.shape[1])
    @property
    def height(self) -> int: return int(self.gray.shape[0])

    # ── Helpers SSS (bounding boxes) ────────────────────────────────────────
    @property
    def has_milco(self) -> bool:
        return any(a.class_id == 1 for a in self.annotations)
    @property
    def has_nombo(self) -> bool:
        return any(a.class_id == 0 for a in self.annotations)

    # ── Helpers FLS (image-level) ────────────────────────────────────────────
    @property
    def is_uxo(self) -> bool:
        return any(a.class_id == 1 for a in self.image_labels)
    @property
    def primary_label(self) -> Optional[ImageLevelAnnotation]:
        """Retorna a primeira anotação de imagem inteira, ou None."""
        return self.image_labels[0] if self.image_labels else None


# ── Leitura de imagem ─────────────────────────────────────────────────────────

def load_image(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega imagem e retorna (bgr, gray).

    Usa read_bytes() + imdecode() para suportar:
      • caminhos Unicode no Windows (acentos, cedilha)
      • formatos PGM, PNG, JPEG sem configuração adicional
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    buf     = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(
            f"OpenCV não conseguiu decodificar: {path}\n"
            "  → Verifique se é JPEG, PNG ou PGM válido e não está corrompido."
        )
    return img_bgr, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ── Parsers de anotação ───────────────────────────────────────────────────────

def load_yolo_annotations(
    label_path: str | Path,
    img_w: int,
    img_h: int,
    class_names: Optional[Dict[int, str]] = None,
) -> List[Annotation]:
    """
    Lê arquivo YOLO (.txt) → lista de Annotation com coordenadas absolutas.

    Tolerante a arquivo inexistente (retorna []) e a linhas malformadas.
    """
    cn    = class_names or CLASS_NAMES
    path  = Path(label_path)
    if not path.exists():
        return []
    anns: List[Annotation] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls             = int(parts[0])
                xc, yc, w, h   = map(float, parts[1:5])
            except ValueError:
                continue
            x1 = int((xc - w/2) * img_w);  y1 = int((yc - h/2) * img_h)
            x2 = int((xc + w/2) * img_w);  y2 = int((yc + h/2) * img_h)
            anns.append(Annotation(
                class_id=cls, class_name=cn.get(cls, f"class_{cls}"),
                x_center_norm=xc, y_center_norm=yc, w_norm=w, h_norm=h,
                x1=max(0, x1), y1=max(0, y1),
                x2=min(img_w, x2), y2=min(img_h, y2),
            ))
    return anns


# ── Construtor principal ──────────────────────────────────────────────────────

def load_sample(
    image_path:  str | Path,
    label_path:  Optional[str | Path] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> SonarSample:
    """
    Carrega imagem SSS + anotações YOLO → SonarSample.

    Caminho padrão para label: mesmo stem da imagem, extensão .txt.
    Usado pelo SSSSonarAdapter e como fallback genérico.
    """
    image_path = Path(image_path)
    if label_path is None:
        label_path = image_path.with_suffix(".txt")
    bgr, gray   = load_image(image_path)
    h, w        = gray.shape
    return SonarSample(
        image_path  = image_path,
        bgr         = bgr,
        gray        = gray,
        annotations = load_yolo_annotations(label_path, w, h, class_names),
    )
