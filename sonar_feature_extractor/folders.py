"""folders.py — Resolução de pastas e orquestração multi-pasta."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from .config import ExtractionConfig
from .pipeline import SonarPipeline

logger = logging.getLogger(__name__)
_EXTS = {".jpg",".jpeg",".JPG",".JPEG",".png",".PNG"}


def collect_images(folder: Path, recursive: bool) -> List[Path]:
    glob = folder.rglob if recursive else folder.glob
    imgs: List[Path] = []
    for ext in _EXTS:
        imgs.extend(glob(f"*{ext}"))
    return sorted(set(imgs))


def resolve_folders(folder: Optional[str], folders: Optional[List[str]],
                    folder_list: Optional[str], recursive: bool) -> List[Path]:
    """Unifica as 3 formas de informar pastas em lista validada sem duplicatas."""
    raw: List[str] = []
    if folder:   raw.append(folder)
    if folders:  raw.extend(folders)
    if folder_list:
        lp = Path(folder_list)
        if not lp.exists():
            raise FileNotFoundError(f"Lista não encontrada: {folder_list}")
        for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                raw.append(s)
    if not raw:
        raise ValueError("Nenhuma pasta informada. Use --folder, --folders ou --folder-list.")

    resolved: List[Path] = []; seen: set = set()
    for p in raw:
        path = Path(p).resolve()
        if not path.exists():    raise FileNotFoundError(f"Pasta não encontrada: {path}")
        if not path.is_dir():    raise NotADirectoryError(f"Não é diretório: {path}")
        if recursive:
            subdirs = {i.parent for e in _EXTS for i in path.rglob(f"*{e}")}
            subdirs.add(path)
            for s in sorted(subdirs):
                if s not in seen: resolved.append(s); seen.add(s)
        else:
            if path not in seen: resolved.append(path); seen.add(path)
    return resolved


def run_multi_folder(folders: List[str], output_csv: str,
                     config: ExtractionConfig, recursive: bool = False,
                     resume: bool = True) -> pd.DataFrame:
    all_images: List[Path] = []
    src_map: Dict[str, str] = {}
    for f in folders:
        fp = Path(f).resolve()
        imgs = collect_images(fp, recursive=recursive)
        logger.info("  %s → %d imagens", fp.name, len(imgs))
        for img in imgs:
            all_images.append(img); src_map[str(img)] = fp.name
    if not all_images:
        raise FileNotFoundError("Nenhuma imagem encontrada.")
    logger.info("Total: %d imagens de %d pasta(s).", len(all_images), len(folders))
    return SonarPipeline(config).run(all_images, output_csv, src_map, resume=resume)
