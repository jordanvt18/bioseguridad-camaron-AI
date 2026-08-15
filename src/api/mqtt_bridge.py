"""
Puente MQTT → WebSocket para ingestión de sensores IoT reales.

Arquitectura de streaming en producción:

    [Sensores IoT (ESP32/Arduino)] --MQTT--> [Broker Mosquitto/EMQX]
                                                    |
                                              paho-mqtt
                                                    v
                                         [MqttBridge (este módulo)]
                                                    |
                                             asyncio.Queue
                                                    v
                                    [FastAPI WebSocket /ws/pond/{id}]
                                                    |
                                                    v
                                         [Dashboard operativo]

Los sensores publican en el topic:

    bioseguridad/sensors/{pond_id}

con payload JSON:

    {"ph": 7.8, "dissolved_oxygen": 5.2, "salinity": 30.1,
     "turbidity": 25.0, "temperature": 29.1, "ammonia": 0.35}

Este módulo es OPCIONAL: requiere `paho-mqtt`. Si no está instalado,
el import falla con un mensaje claro y la API sigue funcionando con
el streamer simulado.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("bioseguridad.mqtt")

TOPIC_PREFIX = "bioseguridad/sensors/"


class MqttBridge:
    """
    Conecta al broker MQTT, suscribe a los topics de sensores y
    reenvía las lecturas a una cola asíncrona consumida por WebSockets.
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        topic_prefix: str = TOPIC_PREFIX,
    ) -> None:
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.topic_prefix = topic_prefix
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._client: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.running = False

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: int) -> None:
        if rc == 0:
            logger.info("MQTT conectado a %s:%s", self.broker, self.port)
            client.subscribe(self.topic_prefix + "+")
        else:
            logger.error("MQTT error de conexión, código %s", rc)

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            pond_id = msg.topic.removeprefix(self.topic_prefix)
            reading = {
                "type": "reading",
                "source": "mqtt",
                "pond_id": pond_id,
                "sensors": payload,
                "outbreak_probability": None,  # el backend la calcula
            }
            if self._loop and not self.queue.full():
                self._loop.call_soon_threadsafe(self.queue.put_nowait, reading)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("MQTT payload inválido en %s: %s", msg.topic, exc)

    async def start(self) -> None:
        """Inicia la conexión al broker (bloqueante, en hilo separado)."""
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError:
            logger.error(
                "paho-mqtt no está instalado. Ejecuta: pip install paho-mqtt"
            )
            return

        self._loop = asyncio.get_running_loop()
        self._client = mqtt.Client()
        if self.username:
            self._client.username_pw_set(self.username, self.password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.connect_async(self.broker, self.port, 60)
        self._client.loop_start()
        self.running = True
        logger.info("Puente MQTT iniciado (topic %s+)", self.topic_prefix)

    async def stop(self) -> None:
        """Detiene la conexión al broker."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        self.running = False

    async def readings(self):
        """Generador asíncrono de lecturas provenientes del broker."""
        while self.running:
            try:
                yield await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue


def bridge_from_env() -> Optional[MqttBridge]:
    """
    Crea un MqttBridge a partir de variables de entorno si MQTT_ENABLED=1.

    Variables:
        MQTT_ENABLED, MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
    """
    if os.getenv("MQTT_ENABLED", "0") != "1":
        return None
    return MqttBridge(
        broker=os.getenv("MQTT_BROKER", "localhost"),
        port=int(os.getenv("MQTT_PORT", "1883")),
        username=os.getenv("MQTT_USERNAME") or None,
        password=os.getenv("MQTT_PASSWORD") or None,
    )
