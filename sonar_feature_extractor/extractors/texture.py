from __future__ import annotations
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from ..registry import BaseImageExtractor, register_image
from ..config import ExtractionConfig
from ..io import SonarSample

@register_image
class GLCMExtractor(BaseImageExtractor):
    """GLCM (Haralick): distingue textura de fundo de objetos artificiais."""
    name = "glcm"
    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        divisor = 256 // config.glcm_levels
        gray_q  = (sample.gray // divisor).astype(np.uint8)
        angles  = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm    = graycomatrix(gray_q, distances=config.glcm_distances, angles=angles,
                               levels=config.glcm_levels, symmetric=True, normed=True)
        feats: dict = {}
        for prop in config.glcm_properties:
            vals = graycoprops(glcm, prop).flatten()
            feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
            feats[f"glcm_{prop}_std"]  = float(np.std(vals))
        return feats
