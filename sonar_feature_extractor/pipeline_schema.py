"""
pipeline_schema.py
==================
Leitura, validação e representação estruturada do arquivo pipeline.json.

Chaves reservadas ignoradas em QUALQUER nível do JSON:
  "_comment", "settings", "$schema", "version"

Formato suportado
-----------------
{
  "_comment": "comentário livre — ignorado",
  "settings": { "mode": "per_object", "workers": 4, ... },

  "sss_sonar": {
    "_comment": "ignorado aqui também",
    "model_tree": {
      "_comment": "ignorado aqui também",
      "statistical": ["basic_stats", "histogram"],
      "texture":     ["glcm", "haar_wavelet"]
    }
  }
}
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Chaves ignoradas em qualquer nível de aninhamento do JSON
_RESERVED_KEYS = {"settings", "$schema", "_comment", "version"}


@dataclass
class ModelSpec:
    """Especificação de um único modelo → um CSV de saída."""
    name:             str
    sensor_type:      str
    extractor_groups: Dict[str, List[str]]

    @property
    def all_extractor_names(self) -> List[str]:
        seen: set  = set()
        result: list = []
        for names in self.extractor_groups.values():
            for n in names:
                if n not in seen:
                    result.append(n)
                    seen.add(n)
        return result

    @property
    def uses_roi(self) -> bool:
        return "roi" in self.all_extractor_names

    @property
    def image_extractor_names(self) -> List[str]:
        return [n for n in self.all_extractor_names if n != "roi"]

    @property
    def output_filename(self) -> str:
        return f"{self.name}.csv"


@dataclass
class SensorSpec:
    """Todos os models de um tipo de sensor."""
    sensor_type: str
    models: Dict[str, ModelSpec] = field(default_factory=dict)

    @property
    def all_extractor_names(self) -> Set[str]:
        names: set = set()
        for m in self.models.values():
            names.update(m.all_extractor_names)
        return names


@dataclass
class GlobalSettings:
    """Configurações globais da seção 'settings'."""
    mode:             Optional[str]  = None
    workers:          Optional[int]  = None
    output_dir:       Optional[str]  = None
    tag_source:       Optional[bool] = None
    checkpoint_every: Optional[int]  = None
    resume:           Optional[bool] = None

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalSettings":
        return cls(
            mode             = d.get("mode"),
            workers          = d.get("workers"),
            output_dir       = d.get("output_dir"),
            tag_source       = d.get("tag_source"),
            checkpoint_every = d.get("checkpoint_every"),
            resume           = d.get("resume"),
        )


@dataclass
class PipelineSpec:
    """Representação completa do pipeline.json."""
    settings:    GlobalSettings
    sensors:     Dict[str, SensorSpec]
    source_path: Optional[Path] = None

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineSpec":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline JSON não encontrado: {path}")
        try:
            raw: dict = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Pipeline JSON inválido em {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Pipeline JSON deve ser um objeto (dict) no nível raiz.")

        settings = GlobalSettings.from_dict(raw.get("settings", {}))
        sensors: Dict[str, SensorSpec] = {}

        for sensor_type, models_data in raw.items():
            # ── Nível 1: ignora chaves reservadas (settings, _comment, etc.) ──
            if sensor_type in _RESERVED_KEYS:
                continue
            if not isinstance(models_data, dict):
                raise ValueError(
                    f"Sensor '{sensor_type}': valor deve ser um objeto com models, "
                    f"recebido: {type(models_data).__name__}"
                )

            spec = SensorSpec(sensor_type=sensor_type)

            for model_name, groups_data in models_data.items():
                # ── Nível 2: ignora _comment dentro do bloco do sensor ─────
                if model_name in _RESERVED_KEYS:
                    continue
                if not isinstance(groups_data, dict):
                    raise ValueError(
                        f"Sensor '{sensor_type}', model '{model_name}': "
                        f"valor deve ser {{grupo: [extractors]}}"
                    )

                extractor_groups: Dict[str, List[str]] = {}

                for group_label, names in groups_data.items():
                    # ── Nível 3: ignora _comment dentro do bloco do model ──
                    if group_label in _RESERVED_KEYS:
                        continue
                    if not isinstance(names, list):
                        raise ValueError(
                            f"Sensor '{sensor_type}', model '{model_name}', "
                            f"grupo '{group_label}': deve ser uma lista de strings. "
                            f"Para comentários use a chave '_comment'."
                        )
                    extractor_groups[group_label] = [str(n) for n in names]

                spec.models[model_name] = ModelSpec(
                    name=model_name, sensor_type=sensor_type,
                    extractor_groups=extractor_groups,
                )

            sensors[sensor_type] = spec

        if not sensors:
            raise ValueError(
                "Pipeline JSON não contém nenhum sensor_type. "
                "Adicione pelo menos 'sss_sonar' com seus models."
            )

        result = cls(settings=settings, sensors=sensors, source_path=path)
        logger.info("Pipeline carregado: %d sensor(es), %d model(s). [%s]",
                    len(sensors), sum(len(s.models) for s in sensors.values()), path.name)
        return result

    def validate(self, available_image: Set[str],
                 available_roi: Set[str], available_sensors: Set[str]) -> None:
        """Valida que todos os nomes no JSON existem nos registries."""
        all_available = available_image | available_roi
        errors: list  = []

        for sensor_type, sensor_spec in self.sensors.items():
            if sensor_type not in available_sensors:
                errors.append(
                    f"Sensor '{sensor_type}' não tem SensorAdapter registrado. "
                    f"Disponíveis: {sorted(available_sensors)}"
                )
            for model_name, model_spec in sensor_spec.models.items():
                for name in model_spec.all_extractor_names:
                    if name not in all_available:
                        errors.append(
                            f"Sensor '{sensor_type}', model '{model_name}': "
                            f"extractor '{name}' não encontrado. "
                            f"Disponíveis: {sorted(all_available)}"
                        )
        if errors:
            raise ValueError("Pipeline JSON inválido:\n" +
                             "\n".join(f"  • {e}" for e in errors))
        logger.info("Pipeline validado com sucesso.")

    def summary(self) -> str:
        lines = ["Pipeline:"]
        for st, ss in self.sensors.items():
            lines.append(f"  [{st}]")
            for mn, ms in ss.models.items():
                lines.append(f"    {mn}.csv ← {ms.all_extractor_names}")
        return "\n".join(lines)
