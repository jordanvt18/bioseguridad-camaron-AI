"""
Streaming de datos en tiempo real para el Simulador de Bioseguridad.

Proporciona:
- SensorStreamer: generador asíncrono que emula un feed en vivo de sensores
  (random walk + episodios de estrés) para piscinas del gestor de datos.
- Endpoints de transporte:
    * WebSocket  /ws/pond/{pond_id}   (bidireccional, recomendado)
    * SSE        /stream/pond/{pond_id} (Server-Sent Events, unidireccional)

En producción, los datos pueden llegar por MQTT (ver mqtt_bridge.py) y
reinyectarse en el mismo formato de mensaje que genera SensorStreamer.
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

# Rangos óptimos por especie (mismos valores que el dashboard)
SPECIES_RANGES: dict[str, dict[str, list[float]]] = {
    "vannamei": {
        "ph": [7.0, 8.5], "dissolved_oxygen": [4.0, 8.0], "salinity": [15.0, 35.0],
        "turbidity": [0.0, 50.0], "temperature": [26.0, 32.0], "ammonia": [0.0, 1.0],
    },
    "stylirostris": {
        "ph": [7.2, 8.6], "dissolved_oxygen": [4.0, 8.0], "salinity": [25.0, 40.0],
        "turbidity": [0.0, 40.0], "temperature": [24.0, 30.0], "ammonia": [0.0, 0.8],
    },
    "monodon": {
        "ph": [7.0, 8.5], "dissolved_oxygen": [4.5, 8.0], "salinity": [15.0, 40.0],
        "turbidity": [0.0, 45.0], "temperature": [27.0, 32.0], "ammonia": [0.0, 0.8],
    },
    "tilapia": {
        "ph": [6.5, 9.0], "dissolved_oxygen": [3.0, 8.0], "salinity": [0.0, 15.0],
        "turbidity": [0.0, 60.0], "temperature": [25.0, 32.0], "ammonia": [0.0, 0.8],
    },
    "trucha": {
        "ph": [6.5, 8.0], "dissolved_oxygen": [6.0, 11.0], "salinity": [0.0, 10.0],
        "turbidity": [0.0, 30.0], "temperature": [10.0, 18.0], "ammonia": [0.0, 0.5],
    },
    "salmon": {
        "ph": [6.8, 8.2], "dissolved_oxygen": [7.0, 12.0], "salinity": [25.0, 35.0],
        "turbidity": [0.0, 25.0], "temperature": [6.0, 14.0], "ammonia": [0.0, 0.4],
    },
    "corvina": {
        "ph": [7.0, 8.5], "dissolved_oxygen": [4.0, 8.0], "salinity": [5.0, 35.0],
        "turbidity": [0.0, 40.0], "temperature": [24.0, 30.0], "ammonia": [0.0, 0.6],
    },
    "atun": {
        "ph": [7.8, 8.3], "dissolved_oxygen": [5.0, 8.0], "salinity": [33.0, 37.0],
        "turbidity": [0.0, 20.0], "temperature": [24.0, 30.0], "ammonia": [0.0, 0.4],
    },
    "dorado": {
        "ph": [7.8, 8.3], "dissolved_oxygen": [5.0, 8.0], "salinity": [15.0, 35.0],
        "turbidity": [0.0, 30.0], "temperature": [24.0, 28.0], "ammonia": [0.0, 0.5],
    },
    "concha_prieta": {
        "ph": [7.6, 8.0], "dissolved_oxygen": [3.5, 4.5], "salinity": [20.0, 28.0],
        "turbidity": [0.0, 60.0], "temperature": [25.0, 28.0], "ammonia": [0.0, 0.8],
    },
}

SENSOR_KEYS = ["ph", "dissolved_oxygen", "salinity", "turbidity", "temperature", "ammonia"]

_DEFAULT_RANGES = SPECIES_RANGES["vannamei"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class SensorStreamer:
    """
    Emula un feed en vivo de sensores para una piscina.

    Los valores evolucionan con un random walk alrededor de los rangos
    óptimos de la especie y ocasionalmente entran en episodios de estrés
    (oxígeno bajo, amonio alto, temperatura alta) que elevan la
    probabilidad de brote.
    """

    def __init__(
        self,
        manager: Any,
        pond_id: str,
        interval: float = 5.0,
        species: str = "vannamei",
        seed: Optional[int] = None,
    ) -> None:
        self.manager = manager
        self.pond_id = pond_id
        self.interval = max(1.0, float(interval))
        self.species = species if species in SPECIES_RANGES else "vannamei"
        self.ranges = SPECIES_RANGES[self.species]
        self._rng = random.Random(seed)
        self._stress_until = 0.0
        self._stress_level = 0.0
        self._base = self._init_base()

    def _init_base(self) -> dict[str, float]:
        """Inicializa los valores base desde el estado actual de la piscina."""
        status = self.manager.get_pond_status(self.pond_id)
        if status is not None and status.sensors is not None:
            s = status.sensors
            return {
                "ph": float(getattr(s, "ph", 7.8) or 7.8),
                "dissolved_oxygen": float(getattr(s, "dissolved_oxygen", 5.5) or 5.5),
                "salinity": float(getattr(s, "salinity", 30.0) or 30.0),
                "turbidity": float(getattr(s, "turbidity", 25.0) or 25.0),
                "temperature": float(getattr(s, "temperature", 29.0) or 29.0),
                "ammonia": float(getattr(s, "ammonia", 0.3) or 0.3),
            }
        r = self.ranges
        return {
            "ph": self._rng.uniform(r["ph"][0], r["ph"][1]),
            "dissolved_oxygen": self._rng.uniform(r["dissolved_oxygen"][0], r["dissolved_oxygen"][1]),
            "salinity": self._rng.uniform(r["salinity"][0], r["salinity"][1]),
            "turbidity": self._rng.uniform(r["turbidity"][0], r["turbidity"][1]),
            "temperature": self._rng.uniform(r["temperature"][0], r["temperature"][1]),
            "ammonia": self._rng.uniform(r["ammonia"][0], r["ammonia"][1] * 0.6),
        }

    def _tick(self) -> dict[str, float]:
        """Avanza un paso el estado de los sensores."""
        r = self.ranges
        now = asyncio.get_event_loop().time()

        # Nuevo episodio de estrés (~4% de probabilidad por tick)
        if now > self._stress_until and self._rng.random() < 0.04:
            self._stress_level = self._rng.uniform(0.4, 1.0)
            self._stress_until = now + self._rng.uniform(20, 90)  # 20-90 s

        active = now < self._stress_until
        stress = self._stress_level if active else 0.0

        b = self._base
        walk = {
            "ph": b["ph"] + self._rng.uniform(-0.05, 0.05) - stress * 0.4,
            "dissolved_oxygen": b["dissolved_oxygen"] + self._rng.uniform(-0.12, 0.12) - stress * 1.4,
            "salinity": b["salinity"] + self._rng.uniform(-0.1, 0.1),
            "turbidity": b["turbidity"] + self._rng.uniform(-0.8, 0.8) + stress * 8.0,
            "temperature": b["temperature"] + self._rng.uniform(-0.08, 0.08) + stress * 0.6,
            "ammonia": b["ammonia"] + self._rng.uniform(-0.01, 0.01) + stress * 0.35,
        }
        # Mantener dentro de rangos físicos amplios
        walk["ph"] = _clamp(walk["ph"], 5.5, 9.5)
        walk["dissolved_oxygen"] = _clamp(walk["dissolved_oxygen"], 0.8, 12.0)
        walk["salinity"] = _clamp(walk["salinity"], 0.0, 45.0)
        walk["turbidity"] = _clamp(walk["turbidity"], 1.0, 200.0)
        walk["temperature"] = _clamp(walk["temperature"], 15.0, 38.0)
        walk["ammonia"] = _clamp(walk["ammonia"], 0.01, 8.0)
        return walk

    def _outbreak_probability(self, sensors: dict[str, float]) -> float:
        """Probabilidad de brote en [0, 1] según desviación de rangos óptimos."""
        r = self.ranges
        score = 0.0
        if sensors["ph"] < r["ph"][0] or sensors["ph"] > r["ph"][1]:
            score += 0.2
        if sensors["dissolved_oxygen"] < r["dissolved_oxygen"][0]:
            score += 0.35
        if sensors["ammonia"] > r["ammonia"][1]:
            score += 0.35
        if sensors["temperature"] > r["temperature"][1] or sensors["temperature"] < r["temperature"][0]:
            score += 0.2
        if sensors["turbidity"] > r["turbidity"][1]:
            score += 0.1
        if sensors["salinity"] > r["salinity"][1] or sensors["salinity"] < r["salinity"][0]:
            score += 0.1
        return _clamp(0.05 + score + self._rng.uniform(-0.04, 0.04), 0.01, 0.98)

    async def readings(self) -> AsyncGenerator[dict[str, Any], None]:
        """Genera lecturas en vivo indefinidamente (dicts JSON-serializables)."""
        while True:
            sensors = self._tick()
            message = {
                "type": "reading",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pond_id": self.pond_id,
                "species": self.species,
                "sensors": sensors,
                "outbreak_probability": round(self._outbreak_probability(sensors), 4),
            }
            yield message
            await asyncio.sleep(self.interval)


def reading_to_json(message: dict[str, Any]) -> str:
    """Serializa un mensaje de lectura a JSON."""
    return json.dumps(message, ensure_ascii=False)


async def sse_event_stream(
    streamer: SensorStreamer,
) -> AsyncGenerator[str, None]:
    """Formatea las lecturas como eventos SSE (text/event-stream)."""
    yield "event: connected\ndata: {\"status\":\"ok\"}\n\n"
    async for message in streamer.readings():
        yield f"event: reading\ndata: {reading_to_json(message)}\n\n"
