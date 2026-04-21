from __future__ import annotations
import numpy as np
from ..registry import BaseImageExtractor, register_image
from ..config import ExtractionConfig
from ..io import SonarSample

@register_image
class ColorChannelExtractor(BaseImageExtractor):
    """Stats por canal BGR da imagem pseudocolorida SSS."""
    name = "color_channels"
    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        feats: dict = {}
        for i, ch in enumerate(["B","G","R"]):
            c = sample.bgr[:,:,i].astype(np.float32)
            feats[f"ch_{ch}_mean"] = float(np.mean(c))
            feats[f"ch_{ch}_std"]  = float(np.std(c))
            feats[f"ch_{ch}_p50"]  = float(np.median(c))
        return feats
