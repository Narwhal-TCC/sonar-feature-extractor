from __future__ import annotations
import cv2
import numpy as np
from skimage.feature import hog
from ..registry import BaseImageExtractor, register_image
from ..config import ExtractionConfig
from ..io import SonarSample

@register_image
class HOGExtractor(BaseImageExtractor):
    """HOG condensado: estrutura direcional das bordas. Útil para SVM."""
    name = "hog"
    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        s   = config.hog_resize
        vec = hog(cv2.resize(sample.gray, (s,s)),
                  orientations=config.hog_orientations,
                  pixels_per_cell=(config.hog_pixels_per_cell, config.hog_pixels_per_cell),
                  cells_per_block=(config.hog_cells_per_block, config.hog_cells_per_block),
                  feature_vector=True)
        return {f"hog_{i:03d}": float(v) for i, v in enumerate(vec)}
