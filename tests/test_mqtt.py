"""
Tests del puente MQTT (lógica de parseo y cola, sin broker real).

Requiere: paho-mqtt (en requirements.txt).
"""

import asyncio

import pytest

from src.api.mqtt_bridge import MqttBridge, bridge_from_env


class FakeMsg:
    def __init__(self, topic: str, payload: bytes):
        self.topic = topic
        self.payload = payload


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.subscribed = None
        self.connected = False

    def subscribe(self, topic):
        self.subscribed = topic

    def connect_async(self, *a, **k):
        self.connected = True

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def username_pw_set(self, *a, **k):
        pass


def test_bridge_from_env_disabled_by_default(monkeypatch):
    """Sin MQTT_ENABLED=1 no se crea puente."""
    monkeypatch.delenv("MQTT_ENABLED", raising=False)
    assert bridge_from_env() is None


def test_bridge_from_env_enabled(monkeypatch):
    """Con MQTT_ENABLED=1 se crea el puente con la config del entorno."""
    monkeypatch.setenv("MQTT_ENABLED", "1")
    monkeypatch.setenv("MQTT_BROKER", "broker.hivemq.com")
    monkeypatch.setenv("MQTT_PORT", "1883")
    bridge = bridge_from_env()
    assert bridge is not None
    assert bridge.broker == "broker.hivemq.com"
    assert bridge.port == 1883


def test_on_message_parses_and_queues():
    """Un mensaje JSON válido entra a la cola con pond_id del topic."""

    async def scenario():
        bridge = MqttBridge()
        bridge._loop = asyncio.get_running_loop()
        payload = b'{"ph": 7.8, "dissolved_oxygen": 5.2, "temperature": 29.1}'
        bridge._on_message(FakeClient(), None, FakeMsg("bioseguridad/sensors/pond-001", payload))
        await asyncio.sleep(0)  # dejar que call_soon_threadsafe ejecute
        assert bridge.queue.qsize() == 1
        reading = bridge.queue.get_nowait()
        assert reading["type"] == "reading"
        assert reading["source"] == "mqtt"
        assert reading["pond_id"] == "pond-001"
        assert reading["sensors"]["ph"] == 7.8

    asyncio.run(scenario())


def test_on_message_invalid_payload_ignored():
    """Payloads inválidos no entran a la cola."""

    async def scenario():
        bridge = MqttBridge()
        bridge._loop = asyncio.get_running_loop()
        bridge._on_message(FakeClient(), None, FakeMsg("bioseguridad/sensors/pond-001", b"no-json"))
        await asyncio.sleep(0)
        assert bridge.queue.qsize() == 0

    asyncio.run(scenario())


def test_readings_generator_yields_queue_items():
    """El generador readings() entrega los mensajes de la cola."""

    async def scenario():
        bridge = MqttBridge()
        bridge.running = True
        bridge._loop = asyncio.get_running_loop()
        bridge._on_message(FakeClient(), None, FakeMsg("bioseguridad/sensors/pond-002", b'{"ph": 8.0}'))
        await asyncio.sleep(0)
        out = []
        async for reading in bridge.readings():
            out.append(reading)
            if len(out) == 1:
                bridge.running = False
                break
        assert len(out) == 1
        assert out[0]["pond_id"] == "pond-002"

    asyncio.run(scenario())


def test_mqtt_client_wiring(monkeypatch):
    """start() configura el cliente paho con topic prefix y credenciales."""
    import paho.mqtt.client as mqtt

    monkeypatch.setattr(mqtt, "Client", FakeClient)

    bridge = MqttBridge(broker="localhost", username="user", password="pass")

    async def run():
        await bridge.start()

    asyncio.run(run())
    assert bridge._client is not None
    assert bridge.running is True
    assert bridge._client.connected is True
