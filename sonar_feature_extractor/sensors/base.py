"""
sensors/base.py — Contrato base para adaptadores de sensor.

Um SensorAdapter encapsula tudo que é específico de um tipo de sensor:
  extensões de arquivo, formato dos labels, classes, etc.

COMO ADICIONAR UM NOVO SENSOR (ex: FLS):
  1. Crie sensors/fls.py herdando BaseSensorAdapter
  2. Defina sensor_type = "fls_sonar"
  3. Implemente load_sample() e, se necessário, get_label_path()
  4. Decore com @register_sensor
  5. Adicione `from . import fls` em sensors/__init__.py
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from ..io import SonarSample


class BaseSensorAdapter(ABC):
    sensor_type:      str           = ""
    image_extensions: tuple[str,...] = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")

    @abstractmethod
    def load_sample(self, image_path: str | Path,
                    label_path: Optional[str | Path] = None) -> SonarSample: ...

    def get_label_path(self, image_path: Path) -> Path:
        return image_path.with_suffix(".txt")

    def collect_images(self, folder: Path, recursive: bool) -> List[Path]:
        images: List[Path] = []
        glob_fn = folder.rglob if recursive else folder.glob
        for ext in self.image_extensions:
            images.extend(glob_fn(f"*{ext}"))
        return sorted(set(images))
