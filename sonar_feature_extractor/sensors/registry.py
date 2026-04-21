from __future__ import annotations
import logging
from typing import Dict, Set, Type
from .base import BaseSensorAdapter

logger = logging.getLogger(__name__)
_SENSOR_REGISTRY: Dict[str, BaseSensorAdapter] = {}


def register_sensor(cls: Type[BaseSensorAdapter]) -> Type[BaseSensorAdapter]:
    if not getattr(cls, "sensor_type", ""):
        raise AttributeError(f"{cls.__name__} deve definir sensor_type.")
    if cls.sensor_type in _SENSOR_REGISTRY:
        raise KeyError(f"SensorAdapter '{cls.sensor_type}' já registrado.")
    _SENSOR_REGISTRY[cls.sensor_type] = cls()
    return cls


def get_sensor_adapter(sensor_type: str) -> BaseSensorAdapter:
    if sensor_type not in _SENSOR_REGISTRY:
        raise KeyError(f"Nenhum adapter para '{sensor_type}'. Disponíveis: {sorted(_SENSOR_REGISTRY)}")
    return _SENSOR_REGISTRY[sensor_type]


def get_available_sensor_types() -> Set[str]:
    return set(_SENSOR_REGISTRY.keys())
