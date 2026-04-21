"""
roi.py — Features da Região do Objeto (bounding box).

obj_highlight_ratio é a feature mais discriminativa de todo o pipeline:
captura a proporção de pixels de highlight dentro do bbox via limiar de Otsu —
assinatura física direta do par highlight/shadow do sonar.
"""
from __future__ import annotations
import cv2, numpy as np
from scipy.stats import skew
from ..registry import BaseROIExtractor, register_roi
from ..config import ExtractionConfig
from ..io import Annotation, SonarSample

@register_roi
class ROIExtractor(BaseROIExtractor):
    name = "roi"

    def extract(self, sample: SonarSample, annotation: Annotation, config: ExtractionConfig) -> dict:
        gray = sample.gray
        x1, y1, x2, y2 = annotation.x1, annotation.y1, annotation.x2, annotation.y2
        roi  = gray[y1:y2, x1:x2].astype(np.float32)
        if roi.size == 0:
            return {}

        pad  = max(int(annotation.width*config.roi_context_padding),
                   int(annotation.height*config.roi_context_padding),
                   config.roi_context_min_pad)
        ctx  = gray[max(0,y1-pad):min(gray.shape[0],y2+pad),
                    max(0,x1-pad):min(gray.shape[1],x2+pad)].astype(np.float32)

        roi_mean = float(np.mean(roi))
        _, thresh = cv2.threshold(roi.astype(np.uint8), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hl_ratio = float(np.sum(thresh>0)/thresh.size) if thresh.size else 0.0

        return {
            "obj_roi_mean":        roi_mean,
            "obj_roi_std":         float(np.std(roi)),
            "obj_roi_min":         float(np.min(roi)),
            "obj_roi_max":         float(np.max(roi)),
            "obj_roi_skewness":    float(skew(roi.flatten())),
            "obj_roi_energy":      float(np.sum(roi**2)/roi.size),
            "obj_highlight_ratio": hl_ratio,
            "obj_local_contrast":  abs(roi_mean - float(np.mean(ctx))),
            "obj_area_pixels":     float(annotation.area_pixels),
            "obj_aspect_ratio":    float(annotation.aspect_ratio),
        }
