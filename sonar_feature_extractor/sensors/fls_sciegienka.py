"""
sensors/fls_sciegienka.py
=========================
SensorAdapter para o dataset FLS UXO Synthetic (Ściegienka & Blachnik, 2024).

Dataset: https://www.kaggle.com/datasets/piotres/front-looking-sonar-uxo
Artigo:  https://doi.org/10.3390/s24185946
DOI:     10.3390/s24185946

Características do dataset
--------------------------
• 69.444 imagens PNG 512×399 px em coordenadas polares
• Geradas por simulação (Gazebo + Project DAVE)
• Sonar multibeam a 900 kHz, altura 1–3 m acima do fundo
• Cada caso tem 3 imagens (frame 1, 2, 3) do mesmo objeto
• TODA A METADATA está codificada no nome do arquivo — não há .txt separado
• Classes: UXO (30%) e nonUXO (70%)

Formato do nome do arquivo
--------------------------
{class}_S{height}_OD{dx}_{dy}_{dz}_OP{px}_{py}_{pz}_OO{rotY}_{rotZ}_{model}_{step}.png

Exemplo:
  UXO_S1.23_OD0.45_0.09_0.09_OP0.0_0.0_0.0_OO0_45_pipe_small_1.png

Campos:
  class     → UXO ou nonUXO
  S{h}      → altura do sonar em metros (float)
  OD{x_y_z} → dimensões do objeto: X (comprimento), Y (diâmetro), Z (altura) em metros
  OP{x_y_z} → posição do centro do objeto no espaço (metros)
  OO{y_z}   → rotação do objeto: Y-axis, Z-axis (graus)
  model     → nome do modelo 3D (ex: pipe_small, box_large, tooBig)
  step      → índice do frame: 1 (3 m), 2 (1.5 m), 3 (0 m de distância)

Estrutura esperada de diretórios
---------------------------------
Qualquer das seguintes convenções é aceita:

  Opção A (split Kaggle oficial):
    data/
    ├── train/UXO/   e   train/nonUXO/
    ├── val/UXO/     e   val/nonUXO/
    └── test/UXO/    e   test/nonUXO/

  Opção B (pasta plana, sem split):
    data/
    ├── UXO_S1.23_...1.png
    └── nonUXO_S2.0_...1.png

  Opção C (uma pasta por classe):
    data/
    ├── UXO/
    └── nonUXO/

O adapter detecta a classe pelo prefixo do nome do arquivo,
independente da estrutura de pastas.
"""
from __future__ import annotations
import os

import re
from pathlib import Path
from typing import Optional

from ..io import ImageLevelAnnotation, SonarSample, load_image
from .base import BaseSensorAdapter
from .registry import register_sensor


# ── Mapeamento de classes ─────────────────────────────────────────────────────

FLS_SYNTHETIC_CLASSES = {0: "nonUXO", 1: "UXO"}

# Regex para extrair todos os campos do nome do arquivo.
# Suporta dois formatos observados no dataset:
#   Formato original: UXO_S1.23_OD0.45_0.09_0.09_OP0.0_0.0_0.0_OO0_45_pipe_small_1
#   Formato novo:     UXO_S1.00_OD0.368_0.196_0.196_OP0.00_4.74_0.06_OO266.00_113.00_UXO-L500-D100_0_noise_20240201-234933
_FILENAME_RE = re.compile(
    r"^(?P<cls>UXO|nonUXO)"
    r"_S(?P<sonar_height>[0-9.]+)"
    r"_OD(?P<od_x>[0-9.]+)_(?P<od_y>[0-9.]+)_(?P<od_z>[0-9.]+)"
    r"_OP(?P<op_x>-?[0-9.]+)_(?P<op_y>-?[0-9.]+)_(?P<op_z>-?[0-9.]+)"
    r"_OO(?P<oo_y>-?[0-9.]+)_(?P<oo_z>-?[0-9.]+)"
    r"_(?P<model>[A-Za-z0-9_-]+?)"
    r"_(?P<step>[0-9]+)"
    r"(?:_noise_[0-9]{8}-[0-9]{6})?$",
    re.IGNORECASE,
)


@register_sensor
class FLSSciegienkaAdapter(BaseSensorAdapter):
    """
    Adapter para FLS UXO Synthetic — Ściegienka & Blachnik (2024).
    sensor_type = "fls_uxo_synthetic"
    """
    sensor_type      = "fls_uxo_synthetic"
    image_extensions = (".png", ".PNG")

    def load_sample(
        self,
        image_path: str | Path,
        label_path: Optional[str | Path] = None,
    ) -> SonarSample:
        """
        Carrega imagem PNG e extrai a anotação inteiramente do nome do arquivo.
        O parâmetro label_path é ignorado (não existe arquivo separado).
        """
        image_path = Path(image_path)
        bgr, gray  = load_image(image_path)

        meta        = self.parse_filename(image_path.stem)
        class_id    = meta.pop("class_id")
        class_name  = FLS_SYNTHETIC_CLASSES.get(class_id, f"class_{class_id}")

        label = ImageLevelAnnotation(
            class_id   = class_id,
            class_name = class_name,
            metadata   = meta,
        )
        return SonarSample(
            image_path   = image_path,
            bgr          = bgr,
            gray         = gray,
            image_labels = [label],
        )

    def get_label_path(self, image_path: Path) -> Path:
        """Não existe arquivo de label separado para este dataset."""
        return Path(os.devnull) if hasattr(os, "devnull") else Path("/dev/null")

    @staticmethod
    def parse_filename(stem: str) -> dict:
        """
        Extrai metadados do stem (nome sem extensão) do arquivo.

        Retorna dict com:
          class_id      : 1=UXO, 0=nonUXO
          sonar_height  : altura do sonar em metros (float)
          obj_length    : dimensão X do objeto (comprimento, metros)
          obj_diameter  : dimensão Y do objeto (diâmetro, metros)
          obj_height_m  : dimensão Z do objeto (metros)
          obj_pos_x/y/z : posição do objeto no espaço (metros)
          obj_rot_y_deg : rotação no eixo Y (graus)
          obj_rot_z_deg : rotação no eixo Z (graus) — orientação no plano do fundo
          model_name    : nome do modelo 3D
          flow_step     : índice do frame: 1 (longe), 2 (meio), 3 (perto)

        Se o nome não corresponder ao padrão, retorna dict com class_id
        inferido pelo prefixo e demais campos NaN.
        """
        m = _FILENAME_RE.match(stem)
        if m:
            g = m.groupdict()
            return {
                "class_id":      1 if g["cls"].upper() == "UXO" else 0,
                "sonar_height":  float(g["sonar_height"]),
                "obj_length":    float(g["od_x"]),
                "obj_diameter":  float(g["od_y"]),
                "obj_height_m":  float(g["od_z"]),
                "obj_pos_x":     float(g["op_x"]),
                "obj_pos_y":     float(g["op_y"]),
                "obj_pos_z":     float(g["op_z"]),
                "obj_rot_y_deg": float(g["oo_y"]),
                "obj_rot_z_deg": float(g["oo_z"]),
                "model_name":    g["model"],
                "flow_step":     int(g["step"]),
            }

        # Fallback: tenta inferir a classe pelo prefixo
        stem_upper = stem.upper()
        if stem_upper.startswith("UXO"):
            class_id = 1
        elif stem_upper.startswith("NONUXO"):
            class_id = 0
        else:
            class_id = -1

        import math
        nan = float("nan")
        return {
            "class_id":      class_id,
            "sonar_height":  nan, "obj_length":  nan, "obj_diameter": nan,
            "obj_height_m":  nan, "obj_pos_x":   nan, "obj_pos_y":    nan,
            "obj_pos_z":     nan, "obj_rot_y_deg": nan, "obj_rot_z_deg": nan,
            "model_name":    "", "flow_step":    -1,
        }
