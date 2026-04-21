from __future__ import annotations
import numpy as np
from ..registry import BaseImageExtractor, register_image
from ..config import ExtractionConfig
from ..io import SonarSample

@register_image
class SpatialGridExtractor(BaseImageExtractor):
    """Grid NxN: captura assimetrias espaciais para localização implícita de objetos."""
    name = "spatial_grid"
    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        g = config.grid_size; h, w = sample.gray.shape; ch, cw = h//g, w//g
        feats: dict = {}
        for r in range(g):
            for c in range(g):
                cell = sample.gray[r*ch:(r+1)*ch, c*cw:(c+1)*cw].astype(np.float32)
                feats[f"grid_r{r}c{c}_mean"] = float(np.mean(cell))
                feats[f"grid_r{r}c{c}_std"]  = float(np.std(cell))
        return feats
