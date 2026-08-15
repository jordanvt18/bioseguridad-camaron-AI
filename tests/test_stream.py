"""
Tests del módulo de streaming (WebSocket + SSE).

Requisitos: httpx (TestClient) — incluido en requirements.txt.
"""

import json
import pytest

from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


def test_websocket_pond_receives_status_then_readings(client):
    """El WebSocket envía estado inicial y luego lecturas en vivo."""
    with client.websocket_connect("/ws/pond/pond-001?interval=1") as ws:
        # Mensaje inicial de estado
        first = ws.receive_json()
        assert first["type"] == "status"
        assert first["pond_id"] == "pond-001"

        # Primera lectura del streamer
        reading = ws.receive_json()
        assert reading["type"] == "reading"
        assert reading["pond_id"] == "pond-001"
        sensors = reading["sensors"]
        for key in ["ph", "dissolved_oxygen", "salinity", "turbidity", "temperature", "ammonia"]:
            assert key in sensors
        assert 0.0 <= reading["outbreak_probability"] <= 1.0


def test_websocket_unknown_pond_closes(client):
    """WebSocket de una piscina inexistente se cierra con código 4404."""
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/pond/pond-999") as ws:
            ws.receive_json()


def test_websocket_respects_species_query(client):
    """El parámetro species se propaga al streamer."""
    with client.websocket_connect("/ws/pond/pond-001?interval=1&species=trucha") as ws:
        ws.receive_json()  # status
        reading = ws.receive_json()
        assert reading["species"] == "trucha"


# ---------------------------------------------------------------------------
# SSE (test de unidad del generador — evita colgar el TestClient con stream infinito)
# ---------------------------------------------------------------------------


def test_sse_event_stream_generator(client):
    """El generador SSE emite evento connected y luego lecturas."""
    import asyncio

    from src.api.main import _get_manager
    from src.api.stream import SensorStreamer, sse_event_stream

    mgr = _get_manager()
    streamer = SensorStreamer(mgr, "pond-001", interval=1, species="vannamei")

    async def collect():
        out = []
        async for chunk in sse_event_stream(streamer):
            out.append(chunk)
            if len(out) == 3:  # connected + 2 readings
                break
        return out

    chunks = asyncio.run(asyncio.wait_for(collect(), timeout=10))
    assert chunks[0].startswith("event: connected")
    assert "data: {" in chunks[0]
    reading_events = [c for c in chunks if c.startswith("event: reading")]
    assert len(reading_events) == 2
    assert "outbreak_probability" in reading_events[0]
    assert reading_events[0].endswith("\n\n")


def test_sse_unknown_pond_404(client):
    """SSE de piscina inexistente devuelve 404 (respuesta finita, no cuelga)."""
    resp = client.get("/stream/pond/pond-999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# SensorStreamer (unidad)
# ---------------------------------------------------------------------------


def test_streamer_readings_are_valid(client):
    """El generador produce lecturas con valores dentro de rangos físicos."""
    from src.api.main import _get_manager
    from src.api.stream import SensorStreamer

    mgr = _get_manager()
    streamer = SensorStreamer(mgr, "pond-001", interval=1, species="vannamei")

    import asyncio

    async def collect():
        out = []
        async for msg in streamer.readings():
            out.append(msg)
            if len(out) == 3:
                break
        return out

    msgs = asyncio.run(collect())
    assert len(msgs) == 3
    for msg in msgs:
        s = msg["sensors"]
        assert 5.5 <= s["ph"] <= 9.5
        assert 0.8 <= s["dissolved_oxygen"] <= 12.0
        assert 0.0 <= s["salinity"] <= 45.0
        assert 1.0 <= s["turbidity"] <= 200.0
        assert 15.0 <= s["temperature"] <= 38.0
        assert 0.01 <= s["ammonia"] <= 8.0
        assert 0.0 <= msg["outbreak_probability"] <= 1.0
