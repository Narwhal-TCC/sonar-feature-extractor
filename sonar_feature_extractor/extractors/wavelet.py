"""
wavelet.py — Transformada Wavelet de Haar 2D multi-resolução.

Diferença fundamental vs. FFT:
  FFT é global: diz QUANTO de alta frequência existe, mas perde ONDE.
  Haar é local: preserva a posição espacial das bordas em cada nível.

Para SSS: o par highlight/shadow é um evento localizado.
  Nível 1 (1/2 res): micro-texturas, ruído de speckle
  Nível 2 (1/4 res): bordas do objeto — escala característica de MILCO  ← mais discriminativo
  Nível 3 (1/8 res): transição highlight/shadow, forma geral             ← mais discriminativo
  Nível 4 (1/16 res): assimetrias de larga escala

12 features × 4 níveis = 48 features total.
Implementação em NumPy puro — sem dependência de pywt.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import kurtosis
from ..registry import BaseImageExtractor, register_image
from ..config import ExtractionConfig
from ..io import SonarSample

_N_LEVELS = 4


def _haar_step(img: np.ndarray):
    """Um passo da DWT Haar 2D → (LL, LH, HL, HH)."""
    L  = (img[:, 0::2] + img[:, 1::2]) * 0.5
    H  = (img[:, 0::2] - img[:, 1::2]) * 0.5
    return ((L[0::2]+L[1::2])*0.5, (L[0::2]-L[1::2])*0.5,
            (H[0::2]+H[1::2])*0.5, (H[0::2]-H[1::2])*0.5)


def _pad_even(img: np.ndarray) -> np.ndarray:
    ph, pw = img.shape[0]%2, img.shape[1]%2
    return np.pad(img, ((0,ph),(0,pw)), mode="edge") if (ph or pw) else img


def _decompose(img: np.ndarray, n: int):
    cur = img.astype(np.float32)
    levels = []
    for _ in range(n):
        cur = _pad_even(cur)
        LL, LH, HL, HH = _haar_step(cur)
        levels.append({"LL":LL,"LH":LH,"HL":HL,"HH":HH})
        cur = LL
    return levels


def _safe_kurtosis(a: np.ndarray) -> float:
    flat = a.flatten()
    return float(kurtosis(flat, fisher=True)) if flat.size >= 4 else 0.0


@register_image
class HaarWaveletExtractor(BaseImageExtractor):
    """
    Wavelet de Haar multi-resolução (4 níveis).
    Captura energia, esparsidade e assimetria direcional das bordas
    em múltiplas escalas — complementa FFT com localização espacial.
    """
    name = "haar_wavelet"

    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        levels = _decompose(sample.gray, _N_LEVELS)
        feats: dict = {}
        for i, b in enumerate(levels, 1):
            LL, LH, HL, HH = b["LL"], b["LH"], b["HL"], b["HH"]
            eLL = float(np.sum(LL**2)); eLH = float(np.sum(LH**2))
            eHL = float(np.sum(HL**2)); eHH = float(np.sum(HH**2))
            tot = eLL + eLH + eHL + eHH + 1e-12
            det = eLH + eHL + eHH
            feats[f"haar_L{i}_LL_energy_ratio"]    = eLL / tot
            feats[f"haar_L{i}_LH_energy_ratio"]    = eLH / tot
            feats[f"haar_L{i}_HL_energy_ratio"]    = eHL / tot
            feats[f"haar_L{i}_HH_energy_ratio"]    = eHH / tot
            feats[f"haar_L{i}_detail_total_ratio"] = det / tot
            feats[f"haar_L{i}_LH_kurtosis"]        = _safe_kurtosis(LH)
            feats[f"haar_L{i}_HL_kurtosis"]        = _safe_kurtosis(HL)
            feats[f"haar_L{i}_HH_kurtosis"]        = _safe_kurtosis(HH)
            feats[f"haar_L{i}_LH_max_abs"]         = float(np.max(np.abs(LH)))
            feats[f"haar_L{i}_HL_max_abs"]         = float(np.max(np.abs(HL)))
            feats[f"haar_L{i}_HH_max_abs"]         = float(np.max(np.abs(HH)))
            feats[f"haar_L{i}_HL_LH_ratio"]        = (eHL+1e-12) / (eLH+1e-12)
        return feats
