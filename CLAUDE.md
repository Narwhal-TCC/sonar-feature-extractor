# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`sonar-feature-extractor` (formerly `sss-feature-extractor`) is a Python CLI pipeline that transforms underwater sonar images + annotation files into numerical CSVs ready for ML model training. It supports multiple sensor types and generates separate CSVs per ML model from a single JSON pipeline config.

## Installation & Running

```bash
# Install (use non-editable on Windows paths with accents — see Known Issues)
pip install .

# Verify
sonar-feature-extractor --list-extractors
sonar-feature-extractor --list-sensors

# Pipeline mode (recommended)
sonar-feature-extractor --pipeline tests/pipeline_full.json --folder ./data/sss/2010/ --output-dir ./outputs/

# Legacy mode (one CSV, all extractors)
sonar-feature-extractor --folder ./data/sss/2010/ --output ./outputs/all_features.csv --mode per_object

# Single image (debug)
sonar-feature-extractor --image ./data/sss/2010/0061_2010.jpg --output ./outputs/single.csv --verbose
```

## Package Layout

```
sonar_feature_extractor/
├── cli.py              ← argparse entry point; --pipeline, --folder, --image, --mode, --workers
├── config.py           ← ExtractionConfig dataclass (all tunable hyperparameters)
├── io.py               ← load_image() via imdecode (Windows-safe), SonarSample, Annotation, ImageLevelAnnotation
├── pipeline.py         ← extract_sample(), SonarPipeline, Checkpoint
├── engine.py           ← PipelineEngine: processes each image ONCE with union of extractors, then filters per model
├── pipeline_schema.py  ← PipelineSpec.from_json() — validates sensor types and extractor names at load time
├── registry.py         ← @register_image / @register_roi decorators + BaseImageExtractor / BaseROIExtractor ABCs
├── folders.py          ← --folder / --folders / --folder-list / --recursive resolution
├── _worker.py          ← Module-level functions for ProcessPoolExecutor (required for Windows spawn pickling)
├── extractors/         ← 11 extractor modules (see below)
└── sensors/            ← 2 sensor adapters + registry
```

## Adding a New Extractor

1. Create `sonar_feature_extractor/extractors/my_extractor.py`
2. Implement `BaseImageExtractor` (or `BaseROIExtractor` for bbox-level features) with `@register_image`
3. Add `from . import my_extractor` to `sonar_feature_extractor/extractors/__init__.py`
4. Add `"my_extractor": ["my_prefix_"]` to `_EXTRACTOR_PREFIXES` in `engine.py`
5. Reference by name in pipeline.json

```python
from sonar_feature_extractor.registry import BaseImageExtractor, register_image
from sonar_feature_extractor.config import ExtractionConfig
from sonar_feature_extractor.io import SonarSample

@register_image
class MyExtractor(BaseImageExtractor):
    name = "my_extractor"
    def extract(self, sample: SonarSample, config: ExtractionConfig) -> dict:
        return {"my_prefix_feature1": float(...)}
```

## Adding a New Sensor Adapter

1. Create `sonar_feature_extractor/sensors/my_sensor.py`
2. Implement `BaseSensorAdapter` with `@register_sensor`
3. Add `from . import my_sensor` to `sonar_feature_extractor/sensors/__init__.py`

```python
from sonar_feature_extractor.sensors.base import BaseSensorAdapter
from sonar_feature_extractor.sensors.registry import register_sensor

@register_sensor
class MySensorAdapter(BaseSensorAdapter):
    sensor_type = "my_sensor"
    image_extensions = (".pgm",)

    def load_sample(self, image_path, label_path=None) -> SonarSample:
        ...
```

## Pipeline JSON Structure

```json
{
  "settings": {
    "mode": "per_object",     // "per_object" | "per_image"
    "workers": 4,
    "output_dir": "./outputs",
    "tag_source": true,
    "checkpoint_every": 50,
    "resume": true
  },
  "sss_sonar": {
    "model_tree": {
      "group_name": ["basic_stats", "glcm", "roi"]
    }
  }
}
```

`_comment` keys are silently ignored at all nesting levels. CLI flags override JSON settings (except `output_dir` uses CLI when not at default `./outputs`).

## Architecture: The Key Efficiency Pattern

`PipelineEngine` processes each image **exactly once** with the UNION of all extractors referenced across all models for a given sensor. Each model's CSV is produced by column filtering (`_filter_columns`) — not re-processing. The column-to-extractor mapping lives in `_EXTRACTOR_PREFIXES` in `engine.py`.

## Data Types

- **`SonarSample`** — container after image load: `bgr`, `gray`, `annotations` (YOLO bboxes for SSS), `image_labels` (image-level for FLS)
- **`Annotation`** — YOLO bbox in both normalized (0–1) and pixel coordinates (`x1,y1,x2,y2`)
- **`ImageLevelAnnotation`** — FLS image-level label with `metadata` dict (from filename or pose CSV)
- **`ExtractionConfig`** — single source of truth for all hyperparameters; passed to every extractor

## Datasets

| `sensor_type` | Folder | Format | Annotation | Classes |
|---|---|---|---|---|
| `sss_sonar` | `data/sss/{year}/` | JPEG 1024×1024 | YOLO `.txt` co-located | `0=NOMBO`, `1=MILCO` |
| `fls_uxo_synthetic` | `data/fls-uxo-synthetic/Fold_*/train/{uxo,nonuxo}/` | PNG 512×399 | Encoded in filename | `0=nonUXO`, `1=UXO` |
| `fls_uxo_aris` | `data/fls-uxo-aris/aris/{session}/` | PGM (128 beams) | `*_frames.csv` + `*_marks.yaml` | Not yet implemented |

## Extractors Reference

| Name in JSON | Features | Column prefix(es) |
|---|---|---|
| `basic_stats` | 13 | `mean std min max p10 p25 p50 p75 p90 skewness kurtosis iqr energy` |
| `histogram` | 32 | `hist_bin_` |
| `glcm` | 12 | `glcm_` |
| `gradient` | 6 | `sobel_` `laplacian_` |
| `frequency` | 4 | `fft_` |
| `spatial_grid` | 32 | `grid_` |
| `hog` | 324 | `hog_` |
| `color_channels` | 9 | `ch_` |
| `haar_wavelet` | 48 | `haar_` |
| `roi` *(bbox only)* | 10 | `obj_` |
| `fls_filename_meta` | 7 | `fls_syn_` |

## Known Issues / Constraints

**Windows path with accents:** `pip install -e .` fails silently when the project path contains non-ASCII characters (e.g., `Área de Trabalho`). Use `pip install .` (non-editable) instead.

**Windows multiprocessing:** Workers must be module-level functions, never closures. All parallel code uses `functools.partial` with functions from `_worker.py`. Never add `ProcessPoolExecutor` with local lambda/closure workers.

**cv2.imread with Unicode paths:** Always load images via `np.frombuffer(path.read_bytes(), np.uint8)` + `cv2.imdecode()`. Never call `cv2.imread(str(path))` directly.

**FLS mode:** When `SonarSample` has only `image_labels` (no `annotations`), the pipeline ignores `config.mode` and always uses `per_image` — this is correct behavior.

**`has_milco`/`has_nombo` for negative images:** Must be explicitly set to `0` in `_build_metadata()` for SSS images with empty annotation files; they are NOT set automatically by the `if sample.annotations:` branch.
