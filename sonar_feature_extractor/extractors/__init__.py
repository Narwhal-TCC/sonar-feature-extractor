"""
Importa todos os extractors para disparar @register_image / @register_roi.

COMO ADICIONAR UMA NOVA FEATURE:
  1. Crie extractors/minha_feature.py
  2. Herde BaseImageExtractor, decore com @register_image
  3. Adicione: from . import minha_feature
"""
from . import stats, histogram, texture, gradient, frequency
from . import spatial, shape, color, wavelet, roi
from . import fls_filename_meta     # FLS Dataset 1: metadados do filename
# from . import fls_pose_meta       # FLS Dataset 2: metadados de pose (em desenvolvimento)
