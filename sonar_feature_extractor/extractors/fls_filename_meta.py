"""
fls_filename_meta.py
====================
Extractor que lê os metadados físicos codificados no nome do arquivo
do dataset FLS UXO Synthetic (Ściegienka & Blachnik, 2024).

Por que é um extractor de imagem?
  Os metadados do filename (altura do sonar, dimensões do objeto, rotação)
  são informações numéricas valiosas para ML — equivalem às coordenadas
  YOLO para o SSS. Este extractor os "normaliza" como features, tornando-os
  disponíveis para qualquer modelo sem lógica especial.

Features geradas (prefixo fls_syn_):
  fls_syn_sonar_height   : altura do sonar acima do fundo (metros)
  fls_syn_obj_length     : comprimento do objeto (eixo X, metros)
  fls_syn_obj_diameter   : diâmetro do objeto (eixo Y, metros)
  fls_syn_obj_aspect     : razão comprimento/diâmetro (discrimina UXO de box)
  fls_syn_obj_rot_z_deg  : orientação no plano horizontal (graus)
  fls_syn_flow_step      : posição temporal do frame (1=longe, 3=perto)
  fls_syn_is_uxo         : label binário 0=nonUXO, 1=UXO (mesma que label)
"""
from __future__ import annotations

import math
from ..config import ExtractionConfig
from ..io import SonarSample
from ..registry import BaseImageExtractor, register_image


@register_image
class FLSFilenameMeta(BaseImageExtractor):
    """
    Lê metadados físicos do nome do arquivo PNG do dataset FLS Synthetic.
    Só produz features se o SonarSample tiver image_labels com metadata.
    Para outros sensores, retorna dict vazio silenciosamente.
    """
    name = "fls_filename_meta"

    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        if not sample.image_labels:
            return {}

        lbl  = sample.image_labels[0]
        meta = lbl.metadata

        # Campos esperados — NaN se ausentes (imagem de outro dataset)
        nan = float("nan")
        sh  = meta.get("sonar_height",  nan)
        ol  = meta.get("obj_length",    nan)
        od  = meta.get("obj_diameter",  nan)
        rz  = meta.get("obj_rot_z_deg", nan)
        fs  = meta.get("flow_step",     nan)

        # Razão aspecto: discrimina UXO cilíndricos (alto) de objetos cúbicos (baixo)
        aspect = (ol / od) if (not math.isnan(ol) and not math.isnan(od)
                               and od > 0) else nan

        return {
            "fls_syn_sonar_height":  sh,
            "fls_syn_obj_length":    ol,
            "fls_syn_obj_diameter":  od,
            "fls_syn_obj_aspect":    aspect,
            "fls_syn_obj_rot_z_deg": rz,
            "fls_syn_flow_step":     float(fs) if (fs is not None and not math.isnan(float(fs))) else nan,
            "fls_syn_is_uxo":        float(lbl.class_id),
        }
