"""
pipeline.py — Orquestrador central: imagem → linhas CSV.
"""
from __future__ import annotations

import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ExtractionConfig
from .io import Annotation, ImageLevelAnnotation, SonarSample, load_sample

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Checkpoint ────────────────────────────────────────────────────────────────

class Checkpoint:
    def __init__(self, csv_path: Path) -> None:
        self.path      = csv_path.with_suffix(".ckpt.json")
        self.processed: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.processed = set(json.loads(
                    self.path.read_text(encoding="utf-8")).get("processed", []))
                logger.info("Checkpoint: %d imagens já processadas.", len(self.processed))
            except Exception:
                self.processed = set()

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"processed": sorted(self.processed)}, indent=2), encoding="utf-8")

    def mark_done(self, p: str | Path) -> None: self.processed.add(str(p))
    def is_done(self, p: str | Path)  -> bool:  return str(p) in self.processed

    def clear(self) -> None:
        self.processed.clear()
        if self.path.exists(): self.path.unlink()


# ── Helpers de extração ───────────────────────────────────────────────────────

def _build_metadata(sample: SonarSample) -> dict:
    """
    Monta metadados comuns para qualquer tipo de sensor.
    Para SSS usa annotations (bounding boxes).
    Para FLS usa image_labels (classificação de imagem inteira).
    """
    base = {
        "image_path": str(sample.image_path),
        "img_width":  sample.width,
        "img_height": sample.height,
    }
    # SSS: bounding boxes
    if sample.annotations:
        base["n_annotations"] = len(sample.annotations)
        base["has_milco"]     = int(sample.has_milco)
        base["has_nombo"]     = int(sample.has_nombo)
    # FLS: image-level labels
    elif sample.image_labels:
        base["n_annotations"] = len(sample.image_labels)
        base["is_uxo"]        = int(sample.is_uxo)
        lbl = sample.primary_label
        if lbl:
            base["image_class_id"]   = lbl.class_id
            base["image_class_name"] = lbl.class_name
            # Achata metadados do filename/pose como colunas
            for k, v in lbl.metadata.items():
                base[f"meta_{k}"] = v
    else:
        base["n_annotations"] = 0
        base["has_milco"]     = 0
        base["has_nombo"]     = 0
    return base


def _run_image_extractors(sample: SonarSample, config: ExtractionConfig,
                           active: Optional[frozenset] = None) -> dict:
    from .extractors import stats, histogram, texture, gradient  # noqa
    from .extractors import frequency, spatial, shape, color, wavelet  # noqa
    try:
        from .extractors import fls_filename_meta  # noqa — FLS Dataset 1
    except ImportError:
        pass
    try:
        from .extractors import fls_pose_meta  # noqa — FLS Dataset 2
    except ImportError:
        pass
    from .registry import get_image_registry
    feats: dict = {}
    for name, ext in get_image_registry().items():
        if active is None or name in active:
            feats.update(ext.safe_extract(sample, config))
    return feats


def _run_roi_extractors(sample: SonarSample, ann: Annotation,
                        config: ExtractionConfig,
                        active: Optional[frozenset] = None) -> dict:
    if active is not None and "roi" not in active:
        return {}
    from .extractors import roi  # noqa
    from .registry import get_roi_registry
    feats: dict = {}
    for name, ext in get_roi_registry().items():
        if active is None or name in active:
            feats.update(ext.safe_extract(sample, ann, config))
    return feats


def _ann_to_row(ann: Annotation) -> dict:
    return {"ann_class_id": ann.class_id, "ann_class_name": ann.class_name,
            "ann_x_center_norm": ann.x_center_norm, "ann_y_center_norm": ann.y_center_norm,
            "ann_w_norm": ann.w_norm, "ann_h_norm": ann.h_norm}


# ── Função principal de extração (picklável para multiprocessing) ─────────────

def extract_sample(
    image_path:        str | Path,
    label_path:        str | Path | None,
    config:            ExtractionConfig,
    source_folder:     Optional[str] = None,
    active_extractors: Optional[frozenset] = None,
    sensor_type:       Optional[str] = None,
) -> List[dict]:
    """
    Extrai features de uma imagem → lista de linhas para o CSV.

    Suporta:
      • SSS (per_object ou per_image via annotations/bounding boxes)
      • FLS (per_image via image_labels)
    """
    # Usa o adapter do sensor correto quando disponível
    if sensor_type:
        try:
            from .sensors import get_sensor_adapter
            adapter = get_sensor_adapter(sensor_type)
            sample  = adapter.load_sample(image_path, label_path)
        except Exception:
            sample = load_sample(image_path, label_path)
    else:
        sample = load_sample(image_path, label_path)
    base   = _build_metadata(sample)
    if config.tag_source and source_folder:
        base["source_folder"] = source_folder
    base.update(_run_image_extractors(sample, config, active_extractors))

    # ── FLS: sem bounding box → sempre per_image ────────────────────────────
    if sample.image_labels and not sample.annotations:
        base["label"] = int(sample.is_uxo)
        return [base]

    # ── SSS per_image ────────────────────────────────────────────────────────
    if config.mode == "per_image":
        if sample.annotations:
            base["ann_xc_mean"]   = float(np.mean([a.x_center_norm for a in sample.annotations]))
            base["ann_yc_mean"]   = float(np.mean([a.y_center_norm for a in sample.annotations]))
            base["ann_w_mean"]    = float(np.mean([a.w_norm        for a in sample.annotations]))
            base["ann_h_mean"]    = float(np.mean([a.h_norm        for a in sample.annotations]))
            base["ann_area_mean"] = float(np.mean([a.w_norm*a.h_norm for a in sample.annotations]))
        else:
            for k in ["ann_xc_mean","ann_yc_mean","ann_w_mean","ann_h_mean","ann_area_mean"]:
                base[k] = np.nan
        base["label"] = int(getattr(sample, "has_milco", 0))
        return [base]

    # ── SSS per_object ───────────────────────────────────────────────────────
    if not sample.annotations:
        base.update({"ann_class_id": -1, "ann_class_name": "negative",
                     "ann_x_center_norm": np.nan, "ann_y_center_norm": np.nan,
                     "ann_w_norm": np.nan, "ann_h_norm": np.nan, "label": 0})
        return [base]

    rows: List[dict] = []
    for ann in sample.annotations:
        row = dict(base)
        row.update(_ann_to_row(ann))
        row.update(_run_roi_extractors(sample, ann, config, active_extractors))
        row["label"] = ann.class_id
        rows.append(row)
    return rows


# ── Pipeline com checkpoint + paralelismo ─────────────────────────────────────

class SonarPipeline:
    def __init__(self, config: ExtractionConfig) -> None:
        config.validate(); self.config = config

    def run(self, image_paths: List[Path], output_csv: str | Path,
            source_folders: Optional[Dict[str, str]] = None,
            resume: bool = True,
            active_extractors: Optional[frozenset] = None) -> pd.DataFrame:

        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        ckpt = Checkpoint(output_csv)
        if not resume: ckpt.clear()

        pending  = [p for p in image_paths if not ckpt.is_done(p)]
        skipped  = len(image_paths) - len(pending)
        all_rows: List[dict] = []

        if skipped:
            logger.info("Checkpoint: %d imagens puladas.", skipped)
            if output_csv.exists():
                all_rows.extend(pd.read_csv(output_csv).to_dict("records"))

        config = self.config; src_map = source_folders or {}
        active = active_extractors; total = len(pending)
        completed = 0; errors: List[str] = []
        logger.info("Processando %d imagens (workers=%d)...", total, config.n_workers)

        from functools import partial
        from ._worker import pipeline_worker
        _task = partial(pipeline_worker, config=config,
                        source_map=src_map, active_extractors=active)

        if config.n_workers > 1 and total > 1:
            with ProcessPoolExecutor(max_workers=config.n_workers) as exe:
                futures = {exe.submit(_task, p): p for p in pending}
                for fut in as_completed(futures):
                    p = futures[fut]; completed += 1
                    try:
                        all_rows.extend(fut.result()); ckpt.mark_done(p)
                    except Exception as exc:
                        msg = f"{p.name}: {exc}"; errors.append(msg)
                        logger.warning("✗ %s", msg)
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
                    msg = f"{p.name}: {exc}"; errors.append(msg)
                    logger.warning("✗ %s", msg)
                    if not config.skip_errors: raise
                if config.checkpoint_every > 0 and completed % config.checkpoint_every == 0:
                    ckpt.save(); SonarPipeline._save_csv(all_rows, output_csv)

        ckpt.save()
        df = SonarPipeline._save_csv(all_rows, output_csv)
        logger.info("Concluído: %d linhas | %d colunas | %d erro(s). CSV: %s",
                    len(df), len(df.columns), len(errors), output_csv.resolve())
        return df

    @staticmethod
    def _save_csv(rows: List[dict], path: Path) -> pd.DataFrame:
        if not rows: return pd.DataFrame()
        df    = pd.DataFrame(rows)
        first = [c for c in ["image_path","source_folder","img_width","img_height",
                              "n_annotations","has_milco","has_nombo",
                              "is_uxo","image_class_id","image_class_name"]
                 if c in df.columns]
        last  = [c for c in ["label"] if c in df.columns]
        mid   = [c for c in df.columns if c not in set(first)|set(last)]
        df    = df[first + mid + last]
        df.to_csv(path, index=False)
        return df
