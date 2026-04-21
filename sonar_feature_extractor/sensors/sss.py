"""
sensors/sss.py — SensorAdapter para Side-Scan Sonar (SSS).

Características:
  - Imagens JPEG pseudocoloradas
  - Labels YOLO (.txt, coordenadas normalizadas 0-1)
  - Classes: 0=NOMBO, 1=MILCO
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from ..io import SonarSample, load_sample as _load_sample
from .base import BaseSensorAdapter
from .registry import register_sensor


@register_sensor
class SSSSonarAdapter(BaseSensorAdapter):
    sensor_type      = "sss_sonar"
    image_extensions = (".jpg", ".jpeg", ".JPG", ".JPEG")

    def load_sample(self, image_path: str | Path,
                    label_path: Optional[str | Path] = None) -> SonarSample:
        return _load_sample(image_path, label_path)
