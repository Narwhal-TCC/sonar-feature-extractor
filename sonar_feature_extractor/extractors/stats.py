from __future__ import annotations
import numpy as np
from scipy.stats import skew, kurtosis
from ..registry import BaseImageExtractor, register_image
from ..config import ExtractionConfig
from ..io import SonarSample

@register_image
class BasicStatsExtractor(BaseImageExtractor):
    """Estatísticas de primeira ordem: brilho, contraste, assimetria."""
    name = "basic_stats"
    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        flat = sample.gray.flatten().astype(np.float32)
        p    = np.percentile(flat, [10, 25, 50, 75, 90])
        return {"mean": float(np.mean(flat)), "std": float(np.std(flat)),
                "min": float(np.min(flat)), "max": float(np.max(flat)),
                "p10": float(p[0]), "p25": float(p[1]), "p50": float(p[2]),
                "p75": float(p[3]), "p90": float(p[4]),
                "skewness": float(skew(flat)), "kurtosis": float(kurtosis(flat)),
                "iqr": float(p[3]-p[1]), "energy": float(np.sum(flat**2)/flat.size)}
