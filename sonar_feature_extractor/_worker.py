"""
_worker.py
==========
Funções de nível de módulo (module-level) para uso com ProcessPoolExecutor.

POR QUE ESTE ARQUIVO EXISTE
----------------------------
No Windows, `multiprocessing` usa o método `spawn`: cada worker é um processo
Python completamente novo que reimporta o módulo principal. Para enviar uma
função a esse processo, o Python usa `pickle` para serializar a referência.

O pickle só consegue serializar funções que estão acessíveis pelo caminho
`module.function_name` — ou seja, funções de nível de módulo.

Funções definidas DENTRO de outros objetos não são picklávéis:
  ❌ lambda
  ❌ def _task dentro de um método (closure local)
  ❌ def _task dentro de __init__

Por isso, os workers de extração vivem aqui, como funções de módulo puras,
e recebem TODOS os seus parâmetros explicitamente (sem capturar nada do escopo
externo).

No Linux/macOS, `fork` copia o processo inteiro na memória e esse problema
não ocorre — mas o código aqui funciona igualmente bem em ambas as plataformas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .config import ExtractionConfig
from .pipeline import extract_sample


def engine_worker(
    image_path:        Path,
    config:            ExtractionConfig,
    source_map:        Dict[str, str],
    active_extractors: frozenset,
    sensor_type:       Optional[str] = None,
) -> List[dict]:
    """
    Worker do PipelineEngine para ProcessPoolExecutor.
    Picklável (função de módulo) — funciona com spawn no Windows.
    """
    label_path = image_path.with_suffix(".txt")
    source     = source_map.get(str(image_path))
    return extract_sample(image_path, label_path, config, source,
                          active_extractors, sensor_type)


def pipeline_worker(
    image_path:        Path,
    config:            ExtractionConfig,
    source_map:        Dict[str, str],
    active_extractors: Optional[frozenset],
    sensor_type:       Optional[str] = None,
) -> List[dict]:
    """
    Worker do SonarPipeline para ProcessPoolExecutor.
    Picklável (função de módulo) — funciona com spawn no Windows.
    """
    label_path = image_path.with_suffix(".txt")
    source     = source_map.get(str(image_path))
    return extract_sample(image_path, label_path, config, source,
                          active_extractors, sensor_type)
