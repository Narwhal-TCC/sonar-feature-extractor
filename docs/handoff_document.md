# sonar-feature-extractor — Documento Técnico de Handoff

> **Propósito deste documento:** fornecer contexto completo do projeto para continuação do desenvolvimento. Contém o que foi construído, como funciona, o que está pendente e os próximos passos detalhados. Pode ser utilizado para contextualizar IAs sobre o projeto.

---

## 1. Visão Geral do Projeto

### O que é

`sonar-feature-extractor` (v3.1.0) é um componente Python instalável via `pip` que transforma imagens de sonar subaquático (JPEG, PNG, PGM) + arquivos de anotação em CSVs numéricos prontos para treinar modelos de Machine Learning.

O componente foi projetado para ser **extensível por design**: adicionar um novo tipo de sensor ou um novo grupo de features não exige modificar código existente — apenas criar um novo arquivo e registrá-lo.

### Problema que resolve

Datasets de sonar subaquático vêm em formatos heterogêneos: diferentes tipos de sensor (SSS, FLS), diferentes formatos de anotação (YOLO .txt, metadados no filename, arquivos de pose JSON), diferentes estruturas de diretório. O componente unifica tudo em um pipeline configurável via JSON que gera múltiplos CSVs específicos por modelo de ML.

### Datasets suportados

| Dataset | Sensor | Anotação | `sensor_type` | Status |
|---|---|---|---|---|
| Santos & Moura (2024) | Side-Scan Sonar | YOLO `.txt` | `sss_sonar` | ✅ Implementado |
| Ściegienka & Blachnik (2024) | FLS sintético (Gazebo) | Nome do arquivo | `fls_uxo_synthetic` | ✅ Implementado |
| Dahn et al. (2024) | FLS real (ARIS Explorer 3000) | Pose 6-DOF JSON/CSV | `fls_uxo_aris` | 🔲 Planejado (não implementado) |

---

## 2. Instalação e Uso Rápido

### Instalação

```bash
# Importante: instalar de um caminho SEM acentos ou caracteres especiais no Windows
# ex: C:\projetos\sss-feature-extractor (NÃO usar "Área de Trabalho" ou similares)
pip install -e .
```

**Atenção Windows:** O modo `-e` (editable) usa um arquivo `.pth` que o Python não consegue processar se o caminho contiver caracteres não-ASCII. Alternativas:
- Mover o projeto para `C:\projetos\` (recomendado para desenvolvimento)
- Usar `pip install .` (sem `-e`) para uso em produção

### Uso básico

```bash
# Listar sensores e extractors disponíveis
sonar-feature-extractor --list-sensors
sonar-feature-extractor --list-extractors

# Gerar pipeline.json de exemplo
sonar-feature-extractor --generate-pipeline meu_pipeline.json

# Modo pipeline (recomendado)
sonar-feature-extractor \
    --pipeline meu_pipeline.json \
    --folder ./sss-data/ \
    --recursive \
    --output-dir ./outputs/ \
    --workers 4

# Modo legado (1 CSV, todos os extractors)
sonar-feature-extractor --folder ./dataset/ --output features.csv
```

### Exemplo de `pipeline.json`

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
      "statistical": ["basic_stats", "histogram"],
      "texture":     ["glcm", "haar_wavelet"],
      "edges":       ["gradient"],
      "roi":         ["roi"]
    },
    "model_regression": {
      "core":      ["basic_stats", "glcm", "gradient"],
      "frequency": ["frequency", "haar_wavelet"]
    }
  },
  "fls_uxo_synthetic": {
    "model_classification": {
      "statistical":  ["basic_stats", "histogram"],
      "texture":      ["glcm", "haar_wavelet"],
      "fls_metadata": ["fls_filename_meta"]
    }
  }
}
```

---

## 3. Estrutura de Arquivos Completa

```
sonar-feature-extractor/                     ← raiz do repositório
│
├── pyproject.toml                           ← build-backend = "setuptools.build_meta"
│                                              entry point: sonar-feature-extractor = "sonar_feature_extractor.cli:main"
│
├── README.md                                ← guia do usuário
├── .gitignore                               ← exclui outputs/, *.csv, data/, *.jpg, etc.
│
├── docs/                                    ← documentação técnica por dataset
│   ├── sss_sonar_santos2024.md             ← SSS dataset (Santos & Moura, 2024)
│   ├── fls_sciegienka2024.md              ← FLS Dataset 1 (Ściegienka & Blachnik, 2024)
│   └── fls_dahn2024.md                    ← FLS Dataset 2 (Dahn et al., 2024) — A CRIAR
│
├── tests/                                   ← pipelines JSON prontos para uso
│   ├── pipeline_full.json                  ← SSS: 2 models, per_object
│   ├── pipeline_minimal.json               ← SSS: smoke test
│   ├── pipeline_per_image.json             ← SSS: per_image, classificação global
│   ├── pipeline_fls_synthetic.json         ← FLS Dataset 1: 2 models
│   └── run_tests.md                        ← comandos e outputs esperados
│
└── sss_feature_extractor/                  ← pacote Python instalável
    ├── __init__.py                         ← API pública (exports)
    ├── __main__.py                         ← python -m sss_feature_extractor
    ├── cli.py                              ← entry point CLI (argparse)
    ├── config.py                           ← ExtractionConfig (dataclass, todos os params)
    ├── io.py                               ← tipos de dados + leitura de disco
    ├── registry.py                         ← BaseImageExtractor + @register_image
    ├── pipeline.py                         ← orquestrador serial/paralelo + Checkpoint
    ├── pipeline_schema.py                  ← leitura e validação do pipeline.json
    ├── engine.py                           ← PipelineEngine (multi-sensor, multi-model)
    ├── folders.py                          ← resolução de pastas (-folder, --folders, etc.)
    ├── _worker.py                          ← workers picklávéis para Windows (spawn)
    │
    ├── extractors/
    │   ├── __init__.py                     ← importa todos (dispara @register_image)
    │   ├── stats.py                        ← basic_stats (13 features)
    │   ├── histogram.py                    ← histogram (32 features)
    │   ├── texture.py                      ← glcm — Haralick (12 features)
    │   ├── gradient.py                     ← gradient — Sobel + Laplacian (6 features)
    │   ├── frequency.py                    ← frequency — FFT 2D (4 features)
    │   ├── spatial.py                      ← spatial_grid — grid 4×4 (32 features)
    │   ├── shape.py                        ← hog — HOG condensado (324 features)
    │   ├── color.py                        ← color_channels — BGR stats (9 features)
    │   ├── wavelet.py                      ← haar_wavelet — Haar 4 níveis (48 features)
    │   ├── roi.py                          ← roi — bounding box features (10 features)
    │   └── fls_filename_meta.py            ← fls_filename_meta — metadados do filename FLS1
    │
    └── sensors/
        ├── __init__.py                     ← importa todos (dispara @register_sensor)
        ├── base.py                         ← BaseSensorAdapter (ABC)
        ├── registry.py                     ← @register_sensor + get_sensor_adapter
        ├── sss.py                          ← SSSSonarAdapter (sensor_type="sss_sonar")
        └── fls_sciegienka.py               ← FLSSciegienkaAdapter (sensor_type="fls_uxo_synthetic")
```

---

## 4. Arquitetura — Como Tudo se Conecta

### Fluxo de execução (modo pipeline)

```
CLI: sonar-feature-extractor --pipeline X --folder Y
         │
         ▼
    cli.py::main()
         │
         ├── PipelineSpec.from_json(X)        ← pipeline_schema.py
         │   └── valida sensor_types e extractor names contra os registries
         │
         ├── resolve_folders(Y)               ← folders.py
         │
         └── PipelineEngine(config).run()     ← engine.py
                  │
                  └── Para cada sensor_type no JSON:
                        │
                        ├── get_sensor_adapter(sensor_type)  ← sensors/registry.py
                        │   └── adapter.collect_images(folders)
                        │
                        ├── Calcula UNIÃO de todos os extractors dos N models
                        │
                        └── _process_images(images, union_extractors)
                                  │
                                  └── Para cada imagem (paralelo via ProcessPoolExecutor):
                                        │
                                        engine_worker(image, config, source_map, active, sensor_type)
                                        ← _worker.py (picklável, funciona no Windows)
                                              │
                                              └── extract_sample()  ← pipeline.py
                                                        │
                                                        ├── adapter.load_sample(image)
                                                        │   ← usa o adapter do sensor correto
                                                        │
                                                        ├── _run_image_extractors(sample, active)
                                                        │   ← itera sobre registry, filtra por active
                                                        │
                                                        └── _run_roi_extractors (se "roi" em active)
                                                            ← só para SSS com bounding boxes

                        └── Para cada model: _filter_columns(full_df, model_spec)
                                └── Salva model_name.csv
```

### Sistema de Registry (extensibilidade)

```python
# Como registrar um novo extractor de imagem:
from sss_feature_extractor.registry import BaseImageExtractor, register_image
from sss_feature_extractor.config import ExtractionConfig
from sss_feature_extractor.io import SonarSample

@register_image
class MeuExtractor(BaseImageExtractor):
    name = "meu_extractor"   # ← nome usado no pipeline.json

    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        return {"minha_feature_1": ..., "minha_feature_2": ...}

# Depois: adicionar "from . import meu_modulo" em extractors/__init__.py
```

```python
# Como registrar um novo sensor:
from sss_feature_extractor.sensors.base import BaseSensorAdapter
from sss_feature_extractor.sensors.registry import register_sensor

@register_sensor
class MeuSensorAdapter(BaseSensorAdapter):
    sensor_type      = "meu_sensor"
    image_extensions = (".png",)

    def load_sample(self, image_path, label_path=None):
        # Retorna SonarSample com image_labels (FLS) ou annotations (SSS com bbox)
        ...

# Depois: adicionar "from . import meu_sensor" em sensors/__init__.py
```

### Tipos de dados — `io.py`

```python
# Para datasets SSS (com bounding box YOLO):
SonarSample(
    image_path   = Path("..."),
    bgr          = np.ndarray,   # (H, W, 3)
    gray         = np.ndarray,   # (H, W)
    annotations  = [Annotation(class_id, class_name, x_center_norm, y_center_norm,
                                w_norm, h_norm, x1, y1, x2, y2)],
    image_labels = [],           # vazio para SSS
)

# Para datasets FLS (classificação de imagem inteira):
SonarSample(
    image_path   = Path("..."),
    bgr          = ...,
    gray         = ...,
    annotations  = [],           # vazio para FLS
    image_labels = [ImageLevelAnnotation(
        class_id   = 1,          # 1=UXO, 0=nonUXO
        class_name = "UXO",
        metadata   = {           # campos específicos do dataset
            "sonar_height": 1.23,
            "obj_length": 0.45,
            "obj_diameter": 0.09,
            "flow_step": 1,
            ...
        }
    )],
)
```

---

## 5. Extractors Disponíveis

### De imagem inteira (operam sobre `sample.gray` ou `sample.bgr`)

| Nome no JSON | Arquivo | Features | Prefixo das colunas | Notas |
|---|---|---|---|---|
| `basic_stats` | `stats.py` | 13 | `mean`, `std`, `p10`..`p90`, `skewness`, `kurtosis`, `iqr`, `energy` | Estatísticas de 1ª ordem |
| `histogram` | `histogram.py` | 32 | `hist_bin_00`..`hist_bin_31` | Histograma de densidade, 32 bins |
| `glcm` | `texture.py` | 12 | `glcm_*` | Haralick: contrast, dissimilarity, homogeneity, energy, correlation, ASM |
| `gradient` | `gradient.py` | 6 | `sobel_*`, `laplacian_*` | Sobel magnitude + Laplaciano (laplacian_var = proxy de nitidez) |
| `frequency` | `frequency.py` | 4 | `fft_*` | FFT 2D: energia por banda (low/mid/high) + log total |
| `spatial_grid` | `spatial.py` | 32 | `grid_r{r}c{c}_*` | Grid 4×4: média/std por célula |
| `hog` | `shape.py` | 324 | `hog_*` | HOG condensado (resize 128×128, cells 32px) |
| `color_channels` | `color.py` | 9 | `ch_B_*`, `ch_G_*`, `ch_R_*` | Stats por canal BGR |
| `haar_wavelet` | `wavelet.py` | 48 | `haar_L1_*`..`haar_L4_*` | Haar 4 níveis: energia, kurtosis, max_abs por sub-banda |
| `fls_filename_meta` | `fls_filename_meta.py` | 7 | `fls_syn_*` | Metadados físicos do filename (FLS Dataset 1) |

### De ROI/bounding box (operam dentro do bbox de cada objeto SSS)

| Nome no JSON | Arquivo | Features | Prefixo | Notas |
|---|---|---|---|---|
| `roi` | `roi.py` | 10 | `obj_*` | Stats da ROI + highlight_ratio + local_contrast + área + aspect_ratio |

**Total máximo: 490 features** (sem `roi`) ou **500 features** (com `roi`)

---

## 6. Sensores Implementados

### `sss_sonar` — Side-Scan Sonar

- **Arquivo:** `sensors/sss.py`
- **Imagens:** JPEG 1024×1024, pseudocoloradas
- **Labels:** YOLO `.txt` (coordenadas normalizadas 0–1)
- **Classes:** `0=NOMBO`, `1=MILCO`
- **Modo:** `per_object` (recomendado) ou `per_image`
- **Dataset:** Santos & Moura (2024), DOI: 10.6084/m9.figshare.24574879.v2

### `fls_uxo_synthetic` — FLS Sintético (Gazebo/DAVE)

- **Arquivo:** `sensors/fls_sciegienka.py`
- **Imagens:** PNG 512×399, coordenadas polares
- **Labels:** Embutidos no nome do arquivo (regex)
- **Classes:** `0=nonUXO`, `1=UXO`
- **Modo:** sempre `per_image` (sem bounding box)
- **Metadados extraídos do filename:** sonar_height, obj_length, obj_diameter, obj_pos_*, obj_rot_*, model_name, flow_step
- **Dataset:** Ściegienka & Blachnik (2024), DOI: 10.3390/s24185946

**Formato do nome do arquivo:**
```
{class}_S{height}_OD{dx}_{dy}_{dz}_OP{px}_{py}_{pz}_OO{rotY}_{rotZ}_{model}_{step}.png
Exemplo: UXO_S1.23_OD0.45_0.09_0.09_OP0.0_0.0_0.0_OO0_45_pipe_small_1.png
```

---

## 7. Configuração — `ExtractionConfig`

```python
@dataclass
class ExtractionConfig:
    # Histograma
    hist_bins: int = 32

    # GLCM
    glcm_levels: int = 64
    glcm_distances: List[int] = [1, 3]
    glcm_properties: List[str] = ["contrast", "dissimilarity", "homogeneity",
                                   "energy", "correlation", "ASM"]
    # Grid
    grid_size: int = 4

    # HOG
    hog_resize: int = 128
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 32
    hog_cells_per_block: int = 2

    # ROI
    roi_context_padding: float = 0.5
    roi_context_min_pad: int = 10

    # Pipeline
    mode: str = "per_object"        # "per_object" ou "per_image"
    n_workers: int = 4
    tag_source: bool = True
    skip_errors: bool = True
    checkpoint_every: int = 50
```

A seção `settings` do `pipeline.json` sobrescreve esses valores. Flags da CLI têm prioridade sobre `settings`.

---

## 8. Bug Crítico Resolvido — Windows `multiprocessing`

**Problema:** No Windows, `multiprocessing` usa o método `spawn` (cria processo do zero). Funções definidas dentro de métodos (closures locais) não são picklávéis e causam o erro:
```
Can't get local object 'PipelineEngine._process_images.<locals>._task'
```

**Solução:** O arquivo `_worker.py` contém funções de nível de módulo (`engine_worker`, `pipeline_worker`) que são passadas via `functools.partial` para o executor:

```python
# ✅ Correto — função de módulo + partial (picklável)
from functools import partial
from ._worker import engine_worker

_task = partial(engine_worker, config=config, source_map=source_map,
                active_extractors=active, sensor_type=sensor_type)
executor.submit(_task, image_path)

# ❌ Errado — closure local (não picklável no Windows)
def _task(p):
    return extract_sample(p, ...)
executor.submit(_task, image_path)
```

**Sempre que adicionar paralelismo:** usar funções de módulo + `partial`, **nunca** closures locais dentro de métodos.

---

## 9. Problema Resolvido — Caminhos com Acentos no Windows

**Problema:** `cv2.imread()` usa a API C do sistema operacional no Windows e falha silenciosamente com caminhos contendo caracteres não-ASCII (acentos, cedilha).

**Solução em `io.py::load_image()`:**
```python
# ✅ Correto — bytes via Python + imdecode (não recebe caminho)
buf     = np.frombuffer(path.read_bytes(), dtype=np.uint8)
img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

# ❌ Errado — falha silenciosamente no Windows com acentos
img_bgr = cv2.imread(str(path))
```

Isso também funciona para PGM e PNG sem configuração adicional.

---

## 10. O que está Faltando / Próximos Passos

### 10.1 — Dataset FLS 2: Dahn et al. (2024) `fls_uxo_aris` [PRIORIDADE ALTA]

**Referência:**
- Título: An Acoustic and Optical Dataset for the Perception of Underwater Unexploded Ordnance (UXO)
- Autores: Dahn, Nikolas et al. (DFKI — German Research Centre for Artificial Intelligence)
- Conferência: OCEANS 2024 — Halifax
- DOI: [10.5281/zenodo.11068046](https://doi.org/10.5281/zenodo.11068046)
- Repositório de código: https://github.com/dfki-ric/uxo-dataset2024
- Licença: BSD 3-Clause

**O que é:**
- ~100 gravações de 3 tipos diferentes de UXO **reais** (não simulados)
- >74.000 frames de sonar ARIS Explorer 3000 (high-frequency imaging sonar, 1.8–3 MHz)
- Dados coletados em 20–21 de setembro de 2023 em ambiente experimental controlado
- Ground truth dos UXOs via modelos 3D fotogramétricos
- Dados de pose 6-DOF (posição + atitude) por frame
- Licença BSD 3-Clause

**Arquivos no Zenodo:**
```
data_export_recordings.7z   ← dataset principal (16.7 GB)
data_export_polar.7z        ← frames em coordenadas polares (16.5 GB)
data_export_3dmodels.7z     ← modelos 3D fotogramétricos (26 MB)
data_export_calibration.7z  ← parâmetros de calibração
data_processed.7z           ← dados brutos extraídos e cortados (61.5 GB)
```

**Diferença fundamental para o Dataset 1:**

| Aspecto | FLS Dataset 1 (Ściegienka) | FLS Dataset 2 (Dahn) |
|---|---|---|
| Origem | Sintético (Gazebo) | Real (experimental) |
| Anotação | Nome do arquivo | Arquivos de pose separados |
| Tipo de label | Classificação binária UXO/nonUXO | Identidade UXO + pose 6-DOF |
| Sonar | Multibeam simulado 900 kHz | ARIS Explorer 3000 (1.8–3 MHz) |
| Imagens | PNG 512×399 (polar) | PGM/PNG cartesiano + polar |
| Tarefa ML | Classificação de imagem | Detecção + reconstrução 3D |

**O que precisa ser implementado:**

**a) `sensors/fls_dahn.py`**
```python
@register_sensor
class FLSDahnAdapter(BaseSensorAdapter):
    sensor_type      = "fls_uxo_aris"
    image_extensions = (".pgm", ".PGM", ".png", ".PNG")

    def load_sample(self, image_path, label_path=None) -> SonarSample:
        # 1. Carrega imagem (PGM funciona com imdecode — já suportado)
        # 2. Encontra o arquivo de pose correspondente
        # 3. Lê pose 6-DOF (JSON ou CSV, verificar repositório GitHub)
        # 4. Cria ImageLevelAnnotation com metadata de pose
        ...

    def get_label_path(self, image_path: Path) -> Path:
        # Descobrir convenção de nome: mesmo stem .json?
        # Ou um único metadata file por gravação?
        # Verificar: https://github.com/dfki-ric/uxo-dataset2024
        ...

    @staticmethod
    def parse_pose_file(path: Path) -> dict:
        # Retorna: {uxo_type_id, uxo_type_name, tx, ty, tz,
        #           roll, pitch, yaw, distance_to_target,
        #           azimuth_angle, elevation_angle}
        ...
```

**b) `extractors/fls_pose_meta.py`**
```python
@register_image
class FLSPoseMeta(BaseImageExtractor):
    name = "fls_pose_meta"  # prefixo de colunas: "fls_aris_"

    def extract(self, sample, config) -> dict:
        if not sample.image_labels:
            return {}
        pose = sample.image_labels[0].metadata
        return {
            "fls_aris_distance_m":    pose.get("distance_to_target", nan),
            "fls_aris_azimuth_deg":   pose.get("azimuth_angle", nan),
            "fls_aris_elevation_deg": pose.get("elevation_angle", nan),
            "fls_aris_uxo_type_id":   pose.get("uxo_type_id", -1),
            "fls_aris_tx":            pose.get("tx", nan),
            "fls_aris_ty":            pose.get("ty", nan),
            "fls_aris_tz":            pose.get("tz", nan),
        }
```

**c) `_EXTRACTOR_PREFIXES` em `engine.py`:**
```python
# Já existe entrada para fls_pose_meta — só precisa ativar:
"fls_pose_meta": ["fls_aris_"],
```

**d) `sensors/__init__.py`:**
```python
# Descomentar:
from . import fls_dahn
```

**e) `extractors/__init__.py`:**
```python
# Descomentar:
from . import fls_pose_meta
```

**f) `docs/fls_dahn2024.md`:**
Criar documentação completa seguindo o modelo de `docs/fls_sciegienka2024.md`.

**g) `tests/pipeline_fls_aris.json`:**
```json
{
  "settings": { "mode": "per_image", "workers": 4 },
  "fls_uxo_aris": {
    "model_classification": {
      "statistical": ["basic_stats", "histogram"],
      "texture":     ["glcm", "haar_wavelet"],
      "edges":       ["gradient"],
      "fls_pose":    ["fls_pose_meta"]
    }
  }
}
```

**AÇÃO NECESSÁRIA ANTES DE IMPLEMENTAR:** Examinar o repositório https://github.com/dfki-ric/uxo-dataset2024 para entender:
1. Formato exato dos arquivos de pose (JSON? CSV? ROS bag?)
2. Estrutura de diretórios interna do `data_export_recordings.7z`
3. Correspondência entre frames de sonar e arquivos de pose
4. Os 3 tipos de UXO e seus identificadores

---

### 10.2 — Rename `sss_feature_extractor` → `sonar_feature_extractor` (v3.1.0)

O pacote foi renomeado de `sss_feature_extractor` para `sonar_feature_extractor` na v3.1.0. Qualquer notebook ou script externo precisa:
1. Substituir `from sss_feature_extractor import ...` → `from sonar_feature_extractor import ...`
2. O comando CLI é agora `sonar-feature-extractor` (era `sss-feature-extractor`)

---

### 10.3 — Testes Automatizados com `pytest` [PRIORIDADE MÉDIA]

Atualmente os testes são manuais (executar pipeline com dados sintéticos e verificar output). Criar:

```
tests/
├── conftest.py              ← fixtures: imagens sintéticas SSS e FLS
├── test_io.py               ← load_image, load_yolo_annotations, ImageLevelAnnotation
├── test_filename_parser.py  ← FLSSciegienkaAdapter.parse_filename (casos de borda)
├── test_extractors.py       ← cada extractor: shape do output, sem NaN, tipos corretos
├── test_pipeline_schema.py  ← _comment ignorado, erros claros para nomes inválidos
├── test_engine.py           ← _filter_columns preserva meta_*, fls_syn_*
└── test_workers.py          ← picklability de engine_worker e pipeline_worker
```

**Casos de teste críticos para `test_filename_parser.py`:**
```python
# Arquivo com nome mínimo válido
"UXO_S1.0_OD0.45_0.09_0.09_OP0_0_0_OO0_0_model_1.png"

# nonUXO com coordenadas negativas
"nonUXO_S2.1_OD1.5_0.05_0.05_OP-0.3_0.1_0.0_OO0_0_box_large_3.png"

# Nome que NÃO bate o regex (fallback deve funcionar)
"UXO_arquivo_sem_formato_padrao.png"
"nonUXO_sem_step.png"
```

---

### 10.4 — Suporte Multi-sensor no mesmo pipeline.json [PRIORIDADE MÉDIA]

Já funciona teoricamente (o engine itera por sensor_type), mas falta teste com SSS e FLS ao mesmo tempo:

```json
{
  "sss_sonar": {
    "model_sss": { "core": ["basic_stats", "glcm", "roi"] }
  },
  "fls_uxo_synthetic": {
    "model_fls": { "core": ["basic_stats", "fls_filename_meta"] }
  }
}
```

Ponto de atenção: `--folders` passa as mesmas pastas para todos os sensores. O adapter de cada sensor filtra pelas suas `image_extensions`. Se as pastas tiverem imagens `.jpg` e `.png` misturadas, cada sensor pega apenas o que é dele.

---

### 10.5 — `_filter_columns` dinâmico via prefixos [MELHORIAS FUTURAS]

Atualmente o mapa `_EXTRACTOR_PREFIXES` em `engine.py` é hardcoded. Para tornar isso completamente automático, cada extractor poderia declarar seus próprios prefixos:

```python
class BasicStatsExtractor(BaseImageExtractor):
    name = "basic_stats"
    column_prefixes = ["mean", "std", "min", "max", "p10", ...]  # NOVO
```

O `engine.py` poderia consultar `extractor.column_prefixes` em vez de usar o dict hardcoded. Isso eliminaria a necessidade de atualizar `_EXTRACTOR_PREFIXES` ao criar novos extractors.

---

### 10.6 — Modo `per_triplet` para FLS Dataset 1 [FUTURO]

O FLS Dataset 1 tem 3 frames por objeto (flow_step 1, 2, 3). O paper sugere que usar os 3 juntos pode melhorar a performance. Um modo `per_triplet` agregaria os 3 frames em uma única linha do CSV.

Isso exigiria:
1. Novo campo `mode = "per_triplet"` em `ExtractionConfig`
2. Lógica no adapter para identificar e agrupar frames do mesmo caso
3. Features de agregação temporal (ex: diferença entre frame 1 e 3)

---

### 10.7 — Documentação de todas as features no projeto [PRIORIDADE BAIXA]

Existe o arquivo `FEATURE_DOCUMENTATION.md` criado mais cedo na conversa (não está no pacote atual). Deve ser movido para `docs/feature_reference.md` e atualizado com as novas colunas do FLS.

---

## 11. Problemas Conhecidos e Armadilhas

### Windows: caminho com acentos no modo editable
- **Sintoma:** `ModuleNotFoundError: No module named 'sss_feature_extractor'` mesmo após `pip install -e .`
- **Causa:** O arquivo `.pth` gerado contém o caminho com `Á` (de `Área`) que o `site.py` do Python não processa
- **Solução:** Mover o projeto para `C:\projetos\` (sem acentos)

### Windows: workers com `--workers > 1`
- **Sintoma:** `Can't get local object '..._task'`
- **Causa:** `multiprocessing` usa `spawn` no Windows; closures locais não são picklávéis
- **Solução:** Já resolvido via `_worker.py` com `functools.partial`
- **ATENÇÃO:** Se adicionar novo código com `ProcessPoolExecutor`, **nunca usar closures locais** — sempre `functools.partial` com função de módulo

### `_comment` no pipeline.json
- **Problema histórico (resolvido):** o parser lançava erro se `_comment` aparecia dentro de grupos de extractors
- **Status:** Resolvido em `pipeline_schema.py` — `_comment` é ignorado em todos os 3 níveis de aninhamento

### FLS: modo `per_object` ignorado
- **Comportamento atual:** se `SonarSample` só tem `image_labels` (sem `annotations`), o pipeline.py ignora o `config.mode` e usa sempre `per_image`
- **Motivo:** correto — não há bounding box para fazer per_object
- **Documentar claramente** no README e no pipeline de teste

---

## 12. Mapeamento da Conversa → Código

Para referência: cada decisão tomada na conversa e onde ela se materializou no código.

| Decisão | Arquivo | Linha/Seção |
|---|---|---|
| `sonar_pipeline` → `sss_feature_extractor` | `pyproject.toml` + todos os `.py` | entry_point + imports |
| Fix `cv2.imread` Unicode Windows | `io.py::load_image()` | `read_bytes() + imdecode` |
| Fix pickle Windows multiprocessing | `_worker.py` | `engine_worker`, `pipeline_worker` |
| `_comment` ignorado no JSON | `pipeline_schema.py` | loops de parsing (3 níveis) |
| `ImageLevelAnnotation` (FLS sem bbox) | `io.py` | nova dataclass |
| Adapter `fls_uxo_synthetic` | `sensors/fls_sciegienka.py` | `FLSSciegienkaAdapter` |
| Parser do filename FLS | `sensors/fls_sciegienka.py` | `parse_filename()`, regex `_FILENAME_RE` |
| Extractor de metadados FLS | `extractors/fls_filename_meta.py` | `FLSFilenameMeta` |
| Preservar colunas `meta_*` e `fls_syn_*` | `engine.py` | `_META_PREFIXES`, `_filter_columns()` |
| `sensor_type` propagado até worker | `pipeline.py`, `_worker.py`, `engine.py` | parâmetro `sensor_type` |

---

## 13. Datasets de Referência Completos

### Dataset SSS — Santos & Moura (2024)
- **DOI:** 10.6084/m9.figshare.24574879.v2
- **Figshare:** https://figshare.com/articles/dataset/Side-scan_sonar_imaging_for_Mine_detection/24574879/2
- **1.170 imagens** JPEG 1024×1024, coletadas 2010–2021 por AUV Teledyne Marine Gavia
- **Classes:** NOMBO (0) e MILCO (1)
- **Anotação:** YOLO `.txt`, um arquivo por imagem, mesmo stem
- **Documentação:** `docs/sss_sonar_santos2024.md`

### Dataset FLS 1 — Ściegienka & Blachnik (2024)
- **DOI:** 10.3390/s24185946
- **Kaggle:** https://www.kaggle.com/datasets/piotres/front-looking-sonar-uxo
- **69.444 imagens** PNG 512×399, geradas por simulação Gazebo + DAVE
- **Classes:** UXO (1, 30%) e nonUXO (0, 70%)
- **Anotação:** embutida no nome do arquivo (sem arquivo separado)
- **Documentação:** `docs/fls_sciegienka2024.md`

### Dataset FLS 2 — Dahn et al. (2024)
- **DOI:** 10.5281/zenodo.11068046
- **Zenodo:** https://zenodo.org/records/11068046
- **GitHub:** https://github.com/dfki-ric/uxo-dataset2024
- **>74.000 frames** do sonar ARIS Explorer 3000, dados reais coletados em laboratório
- **3 tipos de UXO** reais com modelos 3D fotogramétricos
- **Anotação:** pose 6-DOF por frame (formato a verificar no repositório GitHub)
- **94.7 GB total** nos arquivos do Zenodo
- **Documentação:** `docs/fls_dahn2024.md` — **A CRIAR**
- **Status:** 🔲 Planejado — NÃO implementado

---

## 14. Dependências do Pacote

```toml
[project.dependencies]
numpy       >= 1.24
pandas      >= 2.0
opencv-python >= 4.7
scikit-image  >= 0.21
scipy         >= 1.10
```

**Não há dependência de PyWavelets (`pywt`)** — a Wavelet de Haar foi implementada em NumPy puro por ser a mais simples das wavelets (filtros `[1,1]` e `[1,-1]`).

---

## 15. Resumo Executivo para o Claude Code

> **O que é:** componente Python CLI para extração de features de imagens de sonar subaquático (SSS e FLS) em CSVs para ML.
>
> **Estado atual (v3.0.0):** funcionando com 2 datasets (SSS Santos2024 e FLS Ściegienka2024). O pacote se chama `sss_feature_extractor` e o comando é `sss-feature-extractor`.
>
> **O que falta de maior impacto:** implementar o Dataset FLS 2 (Dahn et al., 2024) — `sensor_type="fls_uxo_aris"`. Para isso: criar `sensors/fls_dahn.py`, `extractors/fls_pose_meta.py`, e `docs/fls_dahn2024.md`. Antes de implementar, examinar https://github.com/dfki-ric/uxo-dataset2024 para entender o formato dos arquivos de pose.
>
> **Bug crítico a não repetir:** em Windows, `ProcessPoolExecutor` usa `spawn`. Nunca usar closures locais em código que vai para workers — sempre `functools.partial` com funções de nível de módulo.
>
> **Extensibilidade:** adicionar novo sensor = criar arquivo em `sensors/` + import em `sensors/__init__.py`. Adicionar nova feature = criar arquivo em `extractors/` + import em `extractors/__init__.py` + entrada no dict `_EXTRACTOR_PREFIXES` de `engine.py`.
