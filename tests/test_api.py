"""
Tests para la API de Bioseguridad Inteligente para Camarón.

Usa FastAPI TestClient para probar todos los endpoints principales:
health check, listado de estanques, estado, riesgo, simulación,
recomendación de política y manejo de errores.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Cliente de prueba de FastAPI con ciclo de vida completo."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests del endpoint de health check."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """El endpoint raíz debe retornar status 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_returns_correct_fields(self, client: TestClient) -> None:
        """Debe incluir status, version, model_loaded y n_ponds."""
        data = response_data(client.get("/"))
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "n_ponds" in data

    def test_health_reports_15_ponds(self, client: TestClient) -> None:
        """Debe reportar 15 estanques monitoreados."""
        data = response_data(client.get("/"))
        assert data["n_ponds"] == 15

    def test_health_status_is_operativo(self, client: TestClient) -> None:
        """El estado debe ser 'operativo'."""
        data = response_data(client.get("/"))
        assert data["status"] == "operativo"


# ---------------------------------------------------------------------------
# Listado de estanques
# ---------------------------------------------------------------------------

class TestPondsEndpoint:
    """Tests del endpoint /ponds."""

    def test_ponds_returns_200(self, client: TestClient) -> None:
        """Debe retornar status 200."""
        response = client.get("/ponds")
        assert response.status_code == 200

    def test_ponds_returns_list(self, client: TestClient) -> None:
        """Debe retornar una lista."""
        data = response_data(client.get("/ponds"))
        assert isinstance(data, list)

    def test_ponds_returns_15_items(self, client: TestClient) -> None:
        """Debe retornar exactamente 15 estanques."""
        data = response_data(client.get("/ponds"))
        assert len(data) == 15

    def test_ponds_have_required_fields(self, client: TestClient) -> None:
        """Cada estanque debe tener los campos requeridos."""
        data = response_data(client.get("/ponds"))
        for pond in data:
            assert "pond_id" in pond
            assert "farm_id" in pond
            assert "name" in pond
            assert "sensors" in pond
            assert "alert_level" in pond
            assert "last_update" in pond


# ---------------------------------------------------------------------------
# Estado de estanque
# ---------------------------------------------------------------------------

class TestPondStatusEndpoint:
    """Tests del endpoint /pond/{pond_id}/status."""

    def test_valid_pond_returns_200(self, client: TestClient) -> None:
        """Un estanque válido debe retornar 200."""
        response = client.get("/pond/pond-001/status")
        assert response.status_code == 200

    def test_returns_valid_sensor_data(self, client: TestClient) -> None:
        """Debe retornar lecturas de sensores válidas."""
        data = response_data(client.get("/pond/pond-001/status"))
        sensors = data["sensors"]
        assert 0 <= sensors["ph"] <= 14
        assert 0 <= sensors["dissolved_oxygen"] <= 20
        assert 0 <= sensors["salinity"] <= 50
        assert 0 <= sensors["turbidity"] <= 100
        assert 0 <= sensors["temperature"] <= 45
        assert 0 <= sensors["ammonia"] <= 10
        assert "timestamp" in sensors

    def test_returns_correct_pond_id(self, client: TestClient) -> None:
        """Debe retornar el pond_id correcto."""
        data = response_data(client.get("/pond/pond-005/status"))
        assert data["pond_id"] == "pond-005"

    def test_alert_level_is_valid(self, client: TestClient) -> None:
        """El nivel de alerta debe ser un valor válido."""
        data = response_data(client.get("/pond/pond-001/status"))
        valid_levels = {"normal", "precaución", "alerta", "emergencia"}
        assert data["alert_level"] in valid_levels


# ---------------------------------------------------------------------------
# Riesgo
# ---------------------------------------------------------------------------

class TestRiskEndpoint:
    """Tests del endpoint /pond/{pond_id}/risk."""

    def test_valid_pond_returns_200(self, client: TestClient) -> None:
        """Un estanque válido debe retornar 200."""
        response = client.get("/pond/pond-001/risk")
        assert response.status_code == 200

    def test_probability_in_range(self, client: TestClient) -> None:
        """La probabilidad de brote debe estar en [0, 1]."""
        data = response_data(client.get("/pond/pond-001/risk"))
        prob = data["outbreak_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_level_is_valid(self, client: TestClient) -> None:
        """El nivel de riesgo debe ser válido."""
        data = response_data(client.get("/pond/pond-002/risk"))
        valid_levels = {"bajo", "medio", "alto", "crítico"}
        assert data["risk_level"] in valid_levels

    def test_has_contributing_factors(self, client: TestClient) -> None:
        """Debe incluir factores contribuyentes."""
        data = response_data(client.get("/pond/pond-003/risk"))
        assert "contributing_factors" in data
        assert isinstance(data["contributing_factors"], list)
        assert len(data["contributing_factors"]) > 0

    def test_contributing_factors_have_severity(self, client: TestClient) -> None:
        """Cada factor debe tener severidad en [0, 1]."""
        data = response_data(client.get("/pond/pond-001/risk"))
        for factor in data["contributing_factors"]:
            assert "severity" in factor
            assert 0.0 <= factor["severity"] <= 1.0


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------

class TestSimulateEndpoint:
    """Tests del endpoint POST /pond/{pond_id}/simulate."""

    def test_valid_simulation_returns_200(self, client: TestClient) -> None:
        """Una simulación válida debe retornar 200."""
        response = client.post(
            "/pond/pond-001/simulate",
            json={"action": 1, "duration_days": 30},
        )
        assert response.status_code == 200

    def test_returns_economic_results(self, client: TestClient) -> None:
        """Debe retornar resultados económicos completos."""
        data = response_data(client.post(
            "/pond/pond-001/simulate",
            json={"action": 3, "duration_days": 60},
        ))
        assert "revenue" in data
        assert "costs" in data
        assert "profit" in data
        assert "roi_percentage" in data
        assert "losses_avoided" in data
        assert "net_benefit" in data
        assert "action_name" in data

    def test_action_zero_maintain(self, client: TestClient) -> None:
        """La acción 0 debe ser 'mantener condiciones'."""
        data = response_data(client.post(
            "/pond/pond-001/simulate",
            json={"action": 0, "duration_days": 30},
        ))
        assert data["action_taken"] == 0
        assert "mantener" in data["action_name"].lower()

    def test_invalid_action_returns_422(self, client: TestClient) -> None:
        """Una acción fuera de rango debe retornar 422."""
        response = client.post(
            "/pond/pond-001/simulate",
            json={"action": 5, "duration_days": 30},
        )
        assert response.status_code == 422

    def test_invalid_duration_returns_422(self, client: TestClient) -> None:
        """Una duración de 0 días debe retornar 422."""
        response = client.post(
            "/pond/pond-001/simulate",
            json={"action": 1, "duration_days": 0},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Recomendación de política
# ---------------------------------------------------------------------------

class TestPolicyRecommendationEndpoint:
    """Tests del endpoint /policy/recommendation."""

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        """Una petición válida debe retornar 200."""
        response = client.get(
            "/policy/recommendation",
            params={"pond_id": "pond-001"},
        )
        assert response.status_code == 200

    def test_returns_valid_action(self, client: TestClient) -> None:
        """La acción recomendada debe estar en [0, 4]."""
        data = response_data(client.get(
            "/policy/recommendation",
            params={"pond_id": "pond-001"},
        ))
        assert 0 <= data["recommended_action"] <= 4

    def test_confidence_in_range(self, client: TestClient) -> None:
        """La confianza debe estar en [0, 1]."""
        data = response_data(client.get(
            "/policy/recommendation",
            params={"pond_id": "pond-002"},
        ))
        assert 0.0 <= data["confidence"] <= 1.0

    def test_has_expected_outcome(self, client: TestClient) -> None:
        """Debe incluir un resultado esperado descriptivo."""
        data = response_data(client.get(
            "/policy/recommendation",
            params={"pond_id": "pond-003"},
        ))
        assert "expected_outcome" in data
        assert len(data["expected_outcome"]) > 10

    def test_has_economic_impact(self, client: TestClient) -> None:
        """Debe incluir impacto económico."""
        data = response_data(client.get(
            "/policy/recommendation",
            params={"pond_id": "pond-004"},
        ))
        assert "economic_impact" in data
        assert isinstance(data["economic_impact"], (int, float))


# ---------------------------------------------------------------------------
# Mapa de riesgo de granja
# ---------------------------------------------------------------------------

class TestFarmRiskMapEndpoint:
    """Tests del endpoint /farm/{farm_id}/risk-map."""

    def test_valid_farm_returns_200(self, client: TestClient) -> None:
        """Una granja válida debe retornar 200."""
        response = client.get("/farm/farm-001/risk-map")
        assert response.status_code == 200

    def test_returns_pond_list(self, client: TestClient) -> None:
        """Debe retornar una lista de estanques."""
        data = response_data(client.get("/farm/farm-001/risk-map"))
        assert "ponds" in data
        assert isinstance(data["ponds"], list)
        assert len(data["ponds"]) == 5

    def test_has_overall_risk(self, client: TestClient) -> None:
        """Debe incluir riesgo general de la granja."""
        data = response_data(client.get("/farm/farm-002/risk-map"))
        assert "overall_risk" in data
        valid_levels = {"bajo", "medio", "alto", "crítico"}
        assert data["overall_risk"] in valid_levels


# ---------------------------------------------------------------------------
# Manejo de errores
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests de manejo de errores."""

    def test_invalid_pond_status_returns_404(self, client: TestClient) -> None:
        """Un pond_id inválido en /status debe retornar 404."""
        response = client.get("/pond/pond-999/status")
        assert response.status_code == 404

    def test_invalid_pond_risk_returns_404(self, client: TestClient) -> None:
        """Un pond_id inválido en /risk debe retornar 404."""
        response = client.get("/pond/pond-999/risk")
        assert response.status_code == 404

    def test_invalid_pond_simulate_returns_404(self, client: TestClient) -> None:
        """Un pond_id inválido en /simulate debe retornar 404."""
        response = client.post(
            "/pond/pond-999/simulate",
            json={"action": 1, "duration_days": 30},
        )
        assert response.status_code == 404

    def test_invalid_pond_recommendation_returns_404(self, client: TestClient) -> None:
        """Un pond_id inválido en /policy/recommendation debe retornar 404."""
        response = client.get(
            "/policy/recommendation",
            params={"pond_id": "pond-999"},
        )
        assert response.status_code == 404

    def test_invalid_farm_returns_404(self, client: TestClient) -> None:
        """Un farm_id inválido debe retornar 404."""
        response = client.get("/farm/farm-999/risk-map")
        assert response.status_code == 404

    def test_missing_pond_id_param_returns_422(self, client: TestClient) -> None:
        """Falta el parámetro pond_id en /policy/recommendation debe dar 422."""
        response = client.get("/policy/recommendation")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def response_data(response) -> dict:
    """Extrae el JSON de una respuesta HTTP."""
    return response.json()
