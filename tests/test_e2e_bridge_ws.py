"""
E2E del tramo interno: MqttBridge -> WebSocket.

Valida el camino real de producción (el mismo que usaría un broker MQTT)
inyectando una lectura en la cola del puente y verificando que el cliente
WebSocket la recibe con source=mqtt.

El tramo de red externo (broker real) se valida por separado con
scripts/e2e_mqtt_websocket_test.py cuando hay acceso a un broker.

Nota: MQTT_ENABLED se activa SOLO dentro de cada test vía monkeypatch
(se restaura automáticamente) para no contaminar el resto del suite.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MQTT_ENABLED", "1")
    monkeypatch.setenv("MQTT_BROKER", "localhost")
    monkeypatch.setenv("MQTT_PORT", "1883")
    with TestClient(app) as c:
        yield c


def test_bridge_to_websocket_e2e(client):
    """Una lectura inyectada en el puente llega al WebSocket como source=mqtt."""
    bridge = app.state.mqtt_bridge
    assert bridge is not None, "El puente MQTT debe estar activo con MQTT_ENABLED=1"
    assert bridge.running is True

    with client.websocket_connect("/ws/pond/pond-001") as ws:
        # Mensaje inicial de estado
        first = ws.receive_json()
        assert first["type"] == "status"

        # Simular lo que paho haría al recibir del broker:
        # publicar en bioseguridad/sensors/pond-001
        reading = {
            "type": "reading",
            "source": "mqtt",
            "pond_id": "pond-001",
            "timestamp": "2026-08-14T23:00:00+00:00",
            "species": "vannamei",
            "sensors": {
                "ph": 7.61, "dissolved_oxygen": 3.42, "salinity": 29.9,
                "turbidity": 58.0, "temperature": 30.4, "ammonia": 1.55,
            },
            "outbreak_probability": None,
        }
        bridge.queue.put_nowait(reading)

        # El WebSocket debe reenviar la lectura
        msg = ws.receive_json()
        assert msg["type"] == "reading"
        assert msg["source"] == "mqtt"
        assert msg["pond_id"] == "pond-001"
        assert msg["sensors"]["ph"] == 7.61


def test_bridge_filters_other_ponds(client):
    """Lecturas de otras piscinas no se reenvían a este WebSocket."""
    bridge = app.state.mqtt_bridge
    with client.websocket_connect("/ws/pond/pond-002") as ws:
        ws.receive_json()  # status

        # Lectura para otra piscina: no debe llegar aquí
        bridge.queue.put_nowait({
            "type": "reading", "source": "mqtt", "pond_id": "pond-005",
            "timestamp": "2026-08-14T23:00:00+00:00", "species": None,
            "sensors": {"ph": 8.0}, "outbreak_probability": None,
        })
        # Lectura para esta piscina: sí debe llegar
        bridge.queue.put_nowait({
            "type": "reading", "source": "mqtt", "pond_id": "pond-002",
            "timestamp": "2026-08-14T23:00:00+00:00", "species": "vannamei",
            "sensors": {"ph": 7.9}, "outbreak_probability": 0.2,
        })

        msg = ws.receive_json()
        assert msg["pond_id"] == "pond-002"
        assert msg["sensors"]["ph"] == 7.9
