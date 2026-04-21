"""
Importa todos os SensorAdapters para disparar @register_sensor.

COMO ADICIONAR UM NOVO SENSOR:
  1. Crie sensors/meu_sensor.py herdando BaseSensorAdapter
  2. sensor_type = "meu_tipo"
  3. Implemente load_sample()
  4. Decore com @register_sensor
  5. Adicione: from . import meu_sensor
"""
from . import sss
from . import fls_sciegienka        # FLS Dataset 1: Ściegienka & Blachnik (2024)
# from . import fls_dahn            # FLS Dataset 2: Dahn et al. (2024) — em desenvolvimento

from .base     import BaseSensorAdapter
from .registry import register_sensor, get_sensor_adapter, get_available_sensor_types
