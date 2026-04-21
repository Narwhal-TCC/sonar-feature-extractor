# sonar-feature-extractor

Pipeline de extração de features para imagens de sonar subaquático (SSS, FLS), gerando CSVs numéricos prontos para treinar modelos de Machine Learning.

---

## Datasets suportados

| Sensor | Dataset | Citação | DOI |
|--------|---------|---------|-----|
| **SSS** (Side-Scan Sonar) | Santos & Moura (2024) — *Side-scan sonar imaging for mine-like object detection* | JPEG 1024×1024, labels YOLO | [10.6084/m9.figshare.24574879.v2](https://doi.org/10.6084/m9.figshare.24574879.v2) |
| **FLS Sintético** (Forward-Looking Sonar) | Ściegienka & Blachnik (2024) — *A dataset for underwater object detection and classification using forward-looking sonar* | PNG 512×399, label no nome do arquivo | [10.3390/s24185946](https://doi.org/10.3390/s24185946) |

---

## Instalação

**Pré-requisito:** Python 3.10+

```powershell
# Instalar (use pip install . em caminhos com acentos no Windows)
cd sonar-feature-extractor
pip install .

# Verificar
sonar-feature-extractor --list-extractors
sonar-feature-extractor --list-sensors
```

> **Windows com caminho acentuado (e.g. `Área de Trabalho`):** use `pip install .` (não `-e .`).

---

## Uso rápido

### Modo Pipeline (recomendado)

Gera um CSV por modelo de ML a partir de um JSON de configuração:

```powershell
sonar-feature-extractor `
    --pipeline tests/pipeline_full.json `
    --folders data/sss/2010 data/sss/2015 `
    --output-dir outputs/ `
    --workers 4
```

Saída:
```
outputs/
└── sss_sonar/
    ├── model_tree.csv        <- features para Random Forest / XGBoost
    └── model_regression.csv  <- features para Regressão Logística / SVM
```

### FLS Sintético

```powershell
sonar-feature-extractor `
    --pipeline tests/pipeline_fls_synthetic.json `
    --folder data/fls-uxo-synthetic/Fold_1 `
    --output-dir outputs/
```

### Modo legado (1 CSV, todos os extractors)

```powershell
sonar-feature-extractor `
    --folder data/sss/2010 `
    --output outputs/all_features.csv `
    --mode per_object
```

### Imagem única (debug)

```powershell
sonar-feature-extractor `
    --image data/sss/2010/0061_2010.jpg `
    --output outputs/single.csv `
    --verbose
```

---

## Estrutura do pipeline.json

```json
{
  "settings": {
    "mode": "per_object",
    "workers": 4,
    "output_dir": "./outputs",
    "tag_source": true,
    "checkpoint_every": 50,
    "resume": true
  },
  "sss_sonar": {
    "model_tree": {
      "group_name": ["basic_stats", "glcm", "haar_wavelet", "roi"]
    },
    "model_regression": {
      "core": ["basic_stats", "glcm", "gradient", "frequency", "haar_wavelet"]
    }
  }
}
```

`_comment` é ignorado em qualquer nível. Flags CLI sobrescrevem `settings` do JSON.

---

## Extractors disponíveis (11 no total)

### Imagem inteira

| Nome | Features | Prefixo de colunas |
|------|----------|-------------------|
| `basic_stats` | 13 | `mean std min max p10..p90 skewness kurtosis iqr energy` |
| `histogram` | 32 | `hist_bin_` |
| `glcm` | 12 | `glcm_` |
| `gradient` | 6 | `sobel_` `laplacian_` |
| `frequency` | 4 | `fft_` |
| `spatial_grid` | 32 | `grid_` |
| `hog` | 324 | `hog_` |
| `color_channels` | 9 | `ch_` |
| `haar_wavelet` | 48 | `haar_` |
| `fls_filename_meta` | 7 | `fls_syn_` |

### ROI / Bounding Box (apenas SSS per_object)

| Nome | Features | Prefixo |
|------|----------|---------|
| `roi` | 10 | `obj_` |

**Total: ~490 features** (todos os extractors)

---

## Sensores suportados

| `sensor_type` | Extensões | Anotação | Dataset |
|---------------|-----------|----------|---------|
| `sss_sonar` | `.jpg` `.jpeg` | YOLO `.txt` co-localizado | Santos & Moura 2024 |
| `fls_uxo_synthetic` | `.png` | Codificada no nome do arquivo | Ściegienka & Blachnik 2024 |

---

## Adicionando um novo extractor

```python
# 1. Criar sonar_feature_extractor/extractors/meu_extractor.py
from sonar_feature_extractor.registry import BaseImageExtractor, register_image
from sonar_feature_extractor.config import ExtractionConfig
from sonar_feature_extractor.io import SonarSample

@register_image
class MeuExtractor(BaseImageExtractor):
    name = "meu_extractor"
    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        return {"meu_feat1": float(...)}

# 2. sonar_feature_extractor/extractors/__init__.py — adicionar:
from . import meu_extractor

# 3. sonar_feature_extractor/engine.py — adicionar em _EXTRACTOR_PREFIXES:
"meu_extractor": ["meu_feat"],
```

---

## Adicionando um novo sensor

```python
# 1. Criar sonar_feature_extractor/sensors/meu_sensor.py
from sonar_feature_extractor.sensors.base import BaseSensorAdapter
from sonar_feature_extractor.sensors.registry import register_sensor

@register_sensor
class MeuSensorAdapter(BaseSensorAdapter):
    sensor_type = "meu_sensor"
    image_extensions = (".png",)

    def load_sample(self, image_path, label_path=None) -> SonarSample:
        ...

# 2. sonar_feature_extractor/sensors/__init__.py — adicionar:
from . import meu_sensor
```

---

## API programática

```python
from sonar_feature_extractor import ExtractionConfig, PipelineSpec, PipelineEngine

spec   = PipelineSpec.from_json("tests/pipeline_full.json")
config = ExtractionConfig(n_workers=4)
engine = PipelineEngine(config)

results = engine.run(spec, folders=["data/sss/2010/"], output_dir="outputs/")
# results: {"sss_sonar/model_tree": DataFrame, "sss_sonar/model_regression": DataFrame}
```

---

## Referência de flags CLI

```
sonar-feature-extractor [OPCOES]

MODO PIPELINE
  --pipeline JSON         Arquivo pipeline.json
  --output-dir DIR        Diretorio raiz de saida (default: ./outputs)

ENTRADA
  --image ARQUIVO         Imagem unica
  --folder DIR            Pasta com imagens + labels
  --folders DIR...        Multiplas pastas
  --folder-list ARQ       Arquivo com uma pasta por linha (# = comentario)
  --recursive             Desce em subpastas

SAIDA LEGADO
  --output CSV            CSV de saida
  --mode {per_object,per_image}

PERFORMANCE
  --workers N             Processos paralelos (default: 4)
  --checkpoint-every N    Salva progresso a cada N imagens
  --resume / --no-resume  Retoma ou ignora checkpoint

UTILITARIOS
  --list-extractors       Lista extractors disponiveis
  --list-sensors          Lista sensor_types registrados
  --generate-pipeline ARQ Gera pipeline.json de exemplo
  --verbose / --quiet     Nivel de log
```

---

## Referências

- Santos, F. & Moura, G. (2024). *Side-scan sonar image dataset for mine-like object detection*. Figshare. https://doi.org/10.6084/m9.figshare.24574879.v2
- Ściegienka, P. & Blachnik, M. (2024). *A dataset for underwater object detection and classification using forward-looking sonar*. Sensors, 24(18), 5946. https://doi.org/10.3390/s24185946
