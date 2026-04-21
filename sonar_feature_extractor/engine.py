"""
engine.py — PipelineEngine: executa o pipeline.json completo.

Estratégia central de eficiência
---------------------------------
Para N models do mesmo sensor, cada imagem é processada UMA ÚNICA VEZ
com a UNIÃO de todos os extractors necessários (todos os models juntos).
Os DataFrames por model são derivados por seleção de colunas — O(1).

Estrutura de saída
------------------
{output_dir}/
├── sss_sonar/
│   ├── model_tree.csv        ← colunas de basic_stats + glcm + haar + gradient + roi
│   └── model_regression.csv  ← colunas de basic_stats + glcm + frequency + haar
└── fls_sonar/
    └── model_tree.csv
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import ExtractionConfig
from .pipeline import SonarPipeline, Checkpoint, extract_sample
from .pipeline_schema import ModelSpec, PipelineSpec, SensorSpec
from .registry import get_image_registry, get_roi_registry
from .sensors import get_available_sensor_types, get_sensor_adapter

logger = logging.getLogger(__name__)

# Mapa: nome do extractor → prefixos das colunas que ele gera
_EXTRACTOR_PREFIXES: Dict[str, List[str]] = {
    "basic_stats":    ["mean","std","min","max","p10","p25","p50","p75","p90",
                       "skewness","kurtosis","iqr","energy"],
    "histogram":      ["hist_bin_"],
    "glcm":           ["glcm_"],
    "gradient":       ["sobel_","laplacian_"],
    "frequency":      ["fft_"],
    "spatial_grid":   ["grid_"],
    "hog":            ["hog_"],
    "color_channels": ["ch_"],
    "haar_wavelet":   ["haar_"],
    "roi":            ["obj_"],
    # FLS Dataset 1: metadados do filename
    "fls_filename_meta": ["fls_syn_"],
    # FLS Dataset 2: metadados de pose
    "fls_pose_meta":     ["fls_aris_"],
}

# Colunas sempre preservadas, independente do model_spec
# Prefixos dinâmicos (meta_*, fls_syn_*, etc.) são tratados em _filter_columns
_META_COLS = frozenset([
    # Metadados comuns a todos os sensors
    "image_path", "source_folder", "img_width", "img_height", "n_annotations",
    # SSS
    "has_milco", "has_nombo",
    "ann_class_id", "ann_class_name",
    "ann_x_center_norm", "ann_y_center_norm", "ann_w_norm", "ann_h_norm",
    "ann_xc_mean", "ann_yc_mean", "ann_w_mean", "ann_h_mean", "ann_area_mean",
    # FLS (image-level)
    "is_uxo", "image_class_id", "image_class_name",
    # Label (sempre no final)
    "label",
])

# Prefixos dinâmicos — colunas cujo nome começa com estes prefixos
# são sempre preservadas (metadados de sensores específicos)
_META_PREFIXES: tuple[str, ...] = (
    "meta_",        # metadados brutos do filename/pose (qualquer sensor FLS)
    "fls_syn_",     # features extraídas pelo FLSFilenameMeta extractor
    "fls_aris_",    # features extraídas pelo FLSPoseMeta extractor (Dataset 2)
)


def _filter_columns(full_df: pd.DataFrame, model_spec: ModelSpec) -> pd.DataFrame:
    """
    Filtra o DataFrame completo mantendo as colunas do model_spec.

    Regras de inclusão (em ordem de prioridade):
      1. Colunas em _META_COLS            → sempre incluídas
      2. Colunas com prefixo em _META_PREFIXES → sempre incluídas (metadados FLS, etc.)
      3. Colunas cujo prefixo corresponde a um extractor do model_spec → incluídas
    """
    keep: set = set()
    for col in full_df.columns:
        # Regra 1: metadados fixos
        if col in _META_COLS:
            keep.add(col); continue
        # Regra 2: metadados dinâmicos por prefixo (meta_*, fls_syn_*, etc.)
        if any(col.startswith(pfx) for pfx in _META_PREFIXES):
            keep.add(col); continue
        # Regra 3: features pedidas pelo model_spec
        for ext_name in model_spec.all_extractor_names:
            prefixes = _EXTRACTOR_PREFIXES.get(ext_name, [ext_name])
            if any(col.startswith(pfx) for pfx in prefixes):
                keep.add(col); break

    # Preserva ordem original; garante label no final
    ordered = [c for c in full_df.columns if c in keep]
    if "label" in ordered:
        ordered = [c for c in ordered if c != "label"] + ["label"]
    return full_df[ordered]


def _save_model_csv(df: pd.DataFrame, output_dir: Path,
                    sensor_type: str, model_name: str) -> Path:
    dest = output_dir / sensor_type / f"{model_name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


class PipelineEngine:
    """
    Orquestrador do modo pipeline JSON.

    Uso programático:
        spec    = PipelineSpec.from_json("pipeline.json")
        config  = ExtractionConfig(n_workers=8)
        engine  = PipelineEngine(config)
        results = engine.run(spec, folders=["./data/"], output_dir="./out/")
    """

    def __init__(self, config: ExtractionConfig) -> None:
        config.validate()
        self.config = config

    def run(self, spec: PipelineSpec, folders: List[str],
            output_dir: str | Path, recursive: bool = False,
            resume: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Executa o pipeline completo.

        Retorna Dict["{sensor_type}/{model_name}" → DataFrame].
        """
        import sonar_feature_extractor.extractors  # noqa — dispara @register_image
        import sonar_feature_extractor.sensors     # noqa — dispara @register_sensor

        spec.validate(
            available_image   = set(get_image_registry().keys()),
            available_roi     = set(get_roi_registry().keys()),
            available_sensors = get_available_sensor_types(),
        )
        logger.info("\n%s", spec.summary())

        output_dir = Path(output_dir)
        results: Dict[str, pd.DataFrame] = {}

        for sensor_type, sensor_spec in spec.sensors.items():
            logger.info("━━ Sensor: %s (%d model(s)) ━━",
                        sensor_type, len(sensor_spec.models))
            results.update(self._run_sensor(
                sensor_spec=sensor_spec, folders=folders,
                output_dir=output_dir, recursive=recursive, resume=resume,
            ))

        logger.info("━━ Concluído. %d CSV(s) gerado(s). ━━", len(results))
        for key, df in results.items():
            logger.info("  %s → %d linhas × %d colunas", key, len(df), len(df.columns))
        return results

    def _run_sensor(self, sensor_spec: SensorSpec, folders: List[str],
                    output_dir: Path, recursive: bool, resume: bool) -> Dict[str, pd.DataFrame]:
        adapter    = get_sensor_adapter(sensor_spec.sensor_type)
        all_images: List[Path] = []
        source_map: Dict[str, str] = {}

        for folder_str in folders:
            folder = Path(folder_str).resolve()
            imgs   = adapter.collect_images(folder, recursive=recursive)
            logger.info("  [%s] %s → %d imagens",
                        sensor_spec.sensor_type, folder.name, len(imgs))
            for img in imgs:
                all_images.append(img)
                source_map[str(img)] = folder.name

        if not all_images:
            logger.warning("  Nenhuma imagem para sensor '%s'.", sensor_spec.sensor_type)
            return {}

        # ── Uma única passagem com a UNIÃO de todos os extractors ────
        union = frozenset(sensor_spec.all_extractor_names)
        logger.info("  Extractors na união: %s", sorted(union))

        full_csv = output_dir / sensor_spec.sensor_type / "_full.csv"
        full_csv.parent.mkdir(parents=True, exist_ok=True)

        full_df = self._process_images(
            images=all_images, source_map=source_map,
            active=union, output_csv=full_csv, resume=resume,
            sensor_type=sensor_spec.sensor_type,
        )

        if full_df.empty:
            logger.warning("  Nenhuma feature extraída para '%s'.", sensor_spec.sensor_type)
            return {}

        # ── Gera um CSV por model via filtro de colunas ──────────────
        results: Dict[str, pd.DataFrame] = {}
        for model_name, model_spec in sensor_spec.models.items():
            model_df  = _filter_columns(full_df, model_spec)
            dest      = _save_model_csv(model_df, output_dir,
                                        sensor_spec.sensor_type, model_name)
            key       = f"{sensor_spec.sensor_type}/{model_name}"
            results[key] = model_df
            logger.info("  ✅ %s → %d linhas × %d colunas  [%s]",
                        key, len(model_df), len(model_df.columns), dest)

        # Remove CSV e checkpoint temporários
        for tmp in [full_csv, full_csv.with_suffix(".ckpt.json")]:
            if tmp.exists(): tmp.unlink()

        return results

    def _process_images(self, images: List[Path], source_map: Dict[str, str],
                        active: frozenset, output_csv: Path, resume: bool,
                        sensor_type: Optional[str] = None) -> pd.DataFrame:
        """
        Processa imagens com workers paralelos ou em série.

        O worker (engine_worker) é uma função de nível de módulo, não uma
        closure local — garantia de pickle correto no Windows (spawn).
        Os parâmetros são passados explicitamente via functools.partial para
        que cada chamada receba apenas o image_path variável.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        from ._worker import engine_worker

        config    = self.config
        ckpt      = Checkpoint(output_csv)
        if not resume: ckpt.clear()
        pending   = [p for p in images if not ckpt.is_done(p)]
        skipped   = len(images) - len(pending)
        all_rows: List[dict] = []

        if skipped:
            logger.info("  Checkpoint: %d imagens puladas.", skipped)
            if output_csv.exists():
                all_rows.extend(pd.read_csv(output_csv).to_dict("records"))

        total = len(pending); completed = 0

        # partial fixa config, source_map e active — deixa só image_path livre.
        # partial é picklável porque aponta para uma função de módulo.
        _task = partial(engine_worker,
                        config=config,
                        source_map=source_map,
                        active_extractors=active,
                        sensor_type=sensor_type)

        if config.n_workers > 1 and total > 1:
            with ProcessPoolExecutor(max_workers=config.n_workers) as exe:
                futures = {exe.submit(_task, p): p for p in pending}
                for fut in as_completed(futures):
                    p = futures[fut]; completed += 1
                    try:
                        all_rows.extend(fut.result()); ckpt.mark_done(p)
                    except Exception as exc:
                        logger.warning("  ✗ %s: %s", p.name, exc)
                        if not config.skip_errors:
                            exe.shutdown(wait=False, cancel_futures=True); raise
                    if config.checkpoint_every > 0 and completed % config.checkpoint_every == 0:
                        ckpt.save(); SonarPipeline._save_csv(all_rows, output_csv)
        else:
            for p in pending:
                completed += 1
                try:
                    all_rows.extend(_task(p)); ckpt.mark_done(p)
                except Exception as exc:
                    logger.warning("  ✗ %s: %s", p.name, exc)
                    if not config.skip_errors: raise
                if config.checkpoint_every > 0 and completed % config.checkpoint_every == 0:
                    ckpt.save(); SonarPipeline._save_csv(all_rows, output_csv)

        ckpt.save()
        return SonarPipeline._save_csv(all_rows, output_csv)
