from __future__ import annotations
import numpy as np
from ..registry import BaseImageExtractor, register_image
from ..config import ExtractionConfig
from ..io import SonarSample

@register_image
class FrequencyExtractor(BaseImageExtractor):
    """FFT 2D: distribuição de energia por banda de frequência espacial."""
    name = "frequency"
    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        mag   = np.abs(np.fft.fftshift(np.fft.fft2(sample.gray.astype(np.float32))))
        h, w  = mag.shape; cy, cx = h//2, w//2
        r_low = min(h,w)//8; r_mid = min(h,w)//4
        Y, X  = np.ogrid[:h, :w]
        dist  = np.sqrt((X-cx)**2 + (Y-cy)**2)
        low   = float(np.sum(mag[dist<=r_low]**2))
        mid   = float(np.sum(mag[(dist>r_low)&(dist<=r_mid)]**2))
        high  = float(np.sum(mag[dist>r_mid]**2))
        total = low + mid + high + 1e-9
        return {"fft_low_energy_ratio": low/total, "fft_mid_energy_ratio": mid/total,
                "fft_high_energy_ratio": high/total, "fft_total_log_energy": float(np.log1p(total))}
