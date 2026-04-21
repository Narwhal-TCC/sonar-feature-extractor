"""
config.py — Fonte única de verdade para todos os hiperparâmetros do pipeline.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

CLASS_NAMES: dict[int, str] = {0: "NOMBO", 1: "MILCO"}

@dataclass
class ExtractionConfig:
    """Parâmetros configuráveis. Serializável para JSON."""
    # Histograma
    hist_bins: int = 32
    # GLCM
    glcm_levels: int = 64
    glcm_distances: List[int] = field(default_factory=lambda: [1, 3])
    glcm_properties: List[str] = field(default_factory=lambda: [
        "contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"
    ])
    # Grid
    grid_size: int = 4
    # HOG
    hog_resize: int = 128
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 32
    hog_cells_per_block: int = 2
    # ROI
    roi_context_padding: float = 0.5
    roi_context_min_pad: int = 10
    # Pipeline
    mode: str = "per_object"
    n_workers: int = 4
    tag_source: bool = True
    skip_errors: bool = True
    checkpoint_every: int = 50

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ExtractionConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def validate(self) -> None:
        if self.mode not in ("per_object", "per_image"):
            raise ValueError(f"mode inválido: '{self.mode}'")
        if self.n_workers < 1:
            raise ValueError(f"n_workers deve ser >= 1")
