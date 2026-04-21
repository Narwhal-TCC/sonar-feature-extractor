from __future__ import annotations
import cv2, numpy as np
from skimage.filters import sobel
from ..registry import BaseImageExtractor, register_image
from ..config import ExtractionConfig
from ..io import SonarSample

@register_image
class GradientExtractor(BaseImageExtractor):
    """Sobel + Laplaciano. laplacian_var é proxy de 'há um objeto nítido aqui?'"""
    name = "gradient"
    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        sm  = sobel(sample.gray.astype(np.float32))
        lap = cv2.Laplacian(sample.gray, cv2.CV_64F)
        return {"sobel_mean": float(np.mean(sm)), "sobel_std": float(np.std(sm)),
                "sobel_max": float(np.max(sm)),
                "laplacian_mean": float(np.mean(np.abs(lap))),
                "laplacian_std": float(np.std(lap)), "laplacian_var": float(np.var(lap))}
