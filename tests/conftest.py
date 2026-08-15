"""
Fixtures compartidos del suite de tests.

Garantiza que MQTT_ENABLED no quede activo entre módulos: algunos tests
(streaming) no deben arrancar el puente MQTT real, mientras que los tests
E2E del puente lo activan puntualmente vía monkeypatch.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_mqtt_leak(monkeypatch):
    """Por defecto el puente MQTT está apagado; los tests que lo necesiten lo activan."""
    monkeypatch.delenv("MQTT_ENABLED", raising=False)
