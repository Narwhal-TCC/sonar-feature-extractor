"""
registry.py — Padrão Registry + ABC para extractors de imagem e ROI.

Para ADICIONAR uma feature: crie extractors/minha_feature.py,
herde BaseImageExtractor, decore com @register_image, importe em extractors/__init__.py.
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Dict, Set, Type
import numpy as np
from .config import ExtractionConfig
from .io import Annotation, SonarSample

logger = logging.getLogger(__name__)

_IMAGE_REGISTRY: Dict[str, "BaseImageExtractor"] = {}
_ROI_REGISTRY:   Dict[str, "BaseROIExtractor"]   = {}


def register_image(cls: Type["BaseImageExtractor"]) -> Type["BaseImageExtractor"]:
    if not getattr(cls, "name", ""):
        raise AttributeError(f"{cls.__name__} deve definir `name`.")
    if cls.name in _IMAGE_REGISTRY:
        raise KeyError(f"ImageExtractor '{cls.name}' já registrado.")
    _IMAGE_REGISTRY[cls.name] = cls()
    return cls


def register_roi(cls: Type["BaseROIExtractor"]) -> Type["BaseROIExtractor"]:
    if not getattr(cls, "name", ""):
        raise AttributeError(f"{cls.__name__} deve definir `name`.")
    if cls.name in _ROI_REGISTRY:
        raise KeyError(f"ROIExtractor '{cls.name}' já registrado.")
    _ROI_REGISTRY[cls.name] = cls()
    return cls


def get_image_registry() -> Dict[str, "BaseImageExtractor"]: return dict(_IMAGE_REGISTRY)
def get_roi_registry()   -> Dict[str, "BaseROIExtractor"]:   return dict(_ROI_REGISTRY)


class BaseImageExtractor(ABC):
    name: str = ""

    @abstractmethod
    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict: ...

    def safe_extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        try:
            return self.extract(sample, config)
        except Exception as exc:
            logger.warning("ImageExtractor '%s' falhou em %s: %s", self.name, sample.image_path.name, exc)
            return {}


class BaseROIExtractor(ABC):
    name: str = ""

    @abstractmethod
    def extract(self, sample: SonarSample, annotation: Annotation, config: ExtractionConfig) -> dict: ...

    def safe_extract(self, sample: SonarSample, annotation: Annotation, config: ExtractionConfig) -> dict:
        try:
            return self.extract(sample, annotation, config)
        except Exception as exc:
            logger.warning("ROIExtractor '%s' falhou em %s: %s", self.name, sample.image_path.name, exc)
            return {}
