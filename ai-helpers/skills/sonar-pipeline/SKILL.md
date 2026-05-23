---
name: sonar-pipeline
description: >
  Cheatsheet de engenharia para o pipeline sonar-feature-extractor: 4 passos
  para adicionar extractor, 4 passos para adicionar sensor, 3 constraints
  Windows (com exemplos correto/errado), trabalho pendente. Use ao estender
  ou debugar o pipeline. Trigger: extractor, sensor, registry, engine,
  cv2.imread, multiprocessing, worker, pip install -e.
---

# Sonar Pipeline — Cheatsheet de Engenharia

## 4 passos para adicionar um extractor

```python
# 1. Criar sonar_feature_extractor/extractors/my_extractor.py
from sonar_feature_extractor.registry import BaseImageExtractor, register_image
from sonar_feature_extractor.config import ExtractionConfig
from sonar_feature_extractor.io import SonarSample

@register_image
class MyExtractor(BaseImageExtractor):
    name = "my_extractor"           # nome no pipeline.json

    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        return {"my_prefix_feature1": float(...), "my_prefix_feature2": float(...)}
```

```python
# 2. Em sonar_feature_extractor/extractors/__init__.py
from . import my_extractor  # ← dispara o decorator @register_image
```

```python
# 3. Em sonar_feature_extractor/engine.py, dict _EXTRACTOR_PREFIXES:
_EXTRACTOR_PREFIXES = {
    ...
    "my_extractor": ["my_prefix_"],   # filtragem de colunas por modelo
}
```

```json
// 4. Referenciar no pipeline.json
{
  "sss_sonar": {
    "model_X": {
      "my_group": ["my_extractor"]
    }
  }
}
```

Para extractor de ROI (bbox-level, SSS only): use `BaseROIExtractor` + `@register_roi`.

## 4 passos para adicionar um sensor

```python
# 1. Criar sonar_feature_extractor/sensors/my_sensor.py
from sonar_feature_extractor.sensors.base import BaseSensorAdapter
from sonar_feature_extractor.sensors.registry import register_sensor
from sonar_feature_extractor.io import SonarSample, ImageLevelAnnotation

@register_sensor
class MySensorAdapter(BaseSensorAdapter):
    sensor_type      = "my_sensor"
    image_extensions = (".pgm", ".png")

    def load_sample(self, image_path, label_path=None) -> SonarSample:
        # ler imagem (use load_image de io.py — já trata Unicode)
        # ler labels (formato específico do sensor)
        # construir Annotation (SSS) ou ImageLevelAnnotation (FLS)
        ...

    def get_label_path(self, image_path):
        # retornar Path do arquivo de label, ou None se embutido no filename
        ...
```

```python
# 2. Em sonar_feature_extractor/sensors/__init__.py
from . import my_sensor
```

```python
# 3. Documentação: docs/<sensor>.md (seguir modelo de docs/fls_sciegienka2024.md)
```

```json
// 4. Pipeline de teste: tests/pipeline_<sensor>.json
{
  "settings": { "mode": "per_image", "workers": 4 },
  "my_sensor": { "model_test": { "core": ["basic_stats"] } }
}
```

## 3 constraints Windows (CRÍTICO)

### 1. Instalação: `pip install .` (não `-e`)

```bash
# ✅ Funciona em qualquer caminho
pip install .

# ❌ Quebra em caminhos com acentos (Á, ç, etc.)
pip install -e .
# Sintoma: ModuleNotFoundError: No module named 'sonar_feature_extractor'
# Causa: o site.py do Python não processa o .pth com bytes não-ASCII
```

### 2. Carregamento de imagem: `imdecode` + `read_bytes`

```python
# ✅ Correto — em sonar_feature_extractor/io.py::load_image()
import cv2
import numpy as np
from pathlib import Path

def load_image(path: Path):
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"Failed to decode {path}")
    return img_bgr

# ❌ Errado — falha silenciosa no Windows com Unicode
img_bgr = cv2.imread(str(path))
# Sintoma: img_bgr é None em vez de raise; debug exaustivo
```

### 3. Workers: `functools.partial` com função de módulo

```python
# ✅ Correto — picklável no spawn (Windows)
from functools import partial
from sonar_feature_extractor._worker import engine_worker
from concurrent.futures import ProcessPoolExecutor

_task = partial(engine_worker, config=config, source_map=source_map,
                active_extractors=active, sensor_type=sensor_type)
with ProcessPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(_task, image_paths))

# ❌ Errado — closure local NÃO é picklável no spawn
def my_task(p):
    return extract_sample(p, config, ...)  # captura locals
ex.submit(my_task, image_path)
# Sintoma: Can't get local object 'X.<locals>._task'
```

**Regra:** se você está adicionando código com `ProcessPoolExecutor`, mova a função alvo para `_worker.py` e use `functools.partial`. NUNCA closures locais.

## Trabalho pendente (resumo)

### Alta prioridade
1. **`fls_uxo_aris`** (Dahn et al. 2024)
   - Antes de codar: examinar https://github.com/dfki-ric/uxo-dataset2024
   - Criar `sensors/fls_dahn.py` (FLSDahnAdapter, sensor_type="fls_uxo_aris")
   - Criar `extractors/fls_pose_meta.py` (prefixo `fls_aris_`)
   - Criar `docs/fls_dahn2024.md`
   - Criar `tests/pipeline_fls_aris.json`
   - Editar `sensors/__init__.py`, `extractors/__init__.py`, `engine.py::_EXTRACTOR_PREFIXES`

### Média prioridade
2. **Suite pytest** com fixtures de imagens sintéticas (test_io, test_extractors, test_engine, test_workers).
3. **Multi-sensor end-to-end** — pipeline.json com `sss_sonar` + `fls_uxo_synthetic` simultâneos.

### Baixa prioridade
4. **`_filter_columns` dinâmico**: cada extractor declara `column_prefixes` em vez do dict hardcoded em `engine.py`.
5. **Modo `per_triplet`** para FLS Ściegienka (3 frames por objeto agregados em uma linha).

## Comandos rápidos

```bash
sonar-feature-extractor --list-extractors
sonar-feature-extractor --list-sensors
sonar-feature-extractor --generate-pipeline pipeline.json

# Smoke test
sonar-feature-extractor --pipeline tests/pipeline_minimal.json --folder ./tests/data/sss/ --output-dir ./outputs/

# Pipeline completo
sonar-feature-extractor --pipeline tests/pipeline_full.json --folder ./data/sss/2010/ --output-dir ./outputs/ --workers 4 --verbose
```

## Datasets de referência

| Dataset | DOI | URL |
|---|---|---|
| Santos 2024 (SSS) | 10.6084/m9.figshare.24574879.v2 | https://figshare.com/articles/dataset/24574879 |
| Ściegienka 2024 (FLS sintético) | 10.3390/s24185946 | https://www.kaggle.com/datasets/piotres/front-looking-sonar-uxo |
| Dahn 2024 (FLS real ARIS) | 10.5281/zenodo.11068046 | https://github.com/dfki-ric/uxo-dataset2024 |
