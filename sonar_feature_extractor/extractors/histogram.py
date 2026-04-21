from __future__ import annotations
import numpy as np
from ..registry import BaseImageExtractor, register_image
from ..config import ExtractionConfig
from ..io import SonarSample

@register_image
class HistogramExtractor(BaseImageExtractor):
    """Histograma de densidade de probabilidade. Bins configurável via config.hist_bins."""
    name = "histogram"
    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        hist, _ = np.histogram(sample.gray.flatten(), bins=config.hist_bins, range=(0,256), density=True)
        return {f"hist_bin_{i:02d}": float(v) for i, v in enumerate(hist)}
