"""
sonar_feature_extractor v3.1.0
=====================
Pipeline de extração de features para imagens de Sonar (SSS, FLS...).

Modo pipeline (multi-sensor, multi-model):
    spec    = PipelineSpec.from_json("pipeline.json")
    engine  = PipelineEngine(ExtractionConfig(n_workers=8))
    results = engine.run(spec, folders=["./data/"], output_dir="./out/")

Modo simples (1 CSV):
    config   = ExtractionConfig(n_workers=4, mode="per_object")
    pipeline = SonarPipeline(config)
    df       = pipeline.run(image_paths, output_csv="features.csv")
"""
from .config          import ExtractionConfig
from .io              import SonarSample, Annotation, load_sample
from .pipeline        import SonarPipeline, extract_sample
from .pipeline_schema import PipelineSpec, ModelSpec, SensorSpec, GlobalSettings
from .engine          import PipelineEngine
from .folders         import run_multi_folder, resolve_folders
from .registry        import (BaseImageExtractor, BaseROIExtractor,
                               register_image, register_roi,
                               get_image_registry, get_roi_registry)
from .sensors         import (BaseSensorAdapter, register_sensor,
                               get_sensor_adapter, get_available_sensor_types)

__version__ = "3.1.0"
__all__ = [
    "ExtractionConfig",
    "SonarSample", "Annotation", "load_sample",
    "SonarPipeline", "extract_sample",
    "PipelineSpec", "ModelSpec", "SensorSpec", "GlobalSettings",
    "PipelineEngine",
    "run_multi_folder", "resolve_folders",
    "BaseImageExtractor", "BaseROIExtractor",
    "register_image", "register_roi",
    "get_image_registry", "get_roi_registry",
    "BaseSensorAdapter", "register_sensor",
    "get_sensor_adapter", "get_available_sensor_types",
]
