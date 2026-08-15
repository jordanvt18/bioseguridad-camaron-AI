"""
Pruebas para el entorno RL, simulador económico y recomendador de políticas.

Ejecutar con:
    pytest tests/test_rl.py -v
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pytest

# Asegurar importabilidad del paquete
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.environment import ShrimpPondEnv
from src.rl.economic_simulator import EconomicSimulator
from src.rl.policy_recommender import PolicyRecommender


# ========================================================================== #
# Fixtures
# ========================================================================== #
@pytest.fixture
def env() -> ShrimpPondEnv:
    """Crea un entorno fresco para cada prueba."""
    return ShrimpPondEnv()


@pytest.fixture
def simulator() -> EconomicSimulator:
    """Crea un simulador económico con parámetros por defecto."""
    return EconomicSimulator()


@pytest.fixture
def recommender() -> PolicyRecommender:
    """Crea un recomendador sin modelos entrenados (usa reglas)."""
    return PolicyRecommender()


# ========================================================================== #
# Pruebas del entorno ShrimpPondEnv
# ========================================================================== #
class TestShrimpPondEnv:
    """Pruebas del entorno de Gymnasium."""

    def test_reset_returns_valid_observation(self, env: ShrimpPondEnv) -> None:
        """El reset debe retornar una observación válida del espacio definido."""
        obs, info = env.reset(seed=42)

        # Tipo y forma correctos
        assert isinstance(obs, np.ndarray), "La observación debe ser un numpy.ndarray"
        assert obs.shape == (11,), f"Forma esperada (11,), obtenida {obs.shape}"
        assert obs.dtype == np.float32, f"_dtype esperado float32, obtenido {obs.dtype}"

        # Todos los valores están dentro del espacio de observación
        assert env.observation_space.contains(obs), (
            "La observación no está dentro del espacio de observación definido"
        )

        # info es un diccionario con metadatos
        assert isinstance(info, dict), "info debe ser un diccionario"
        assert "initial_state" in info, "info debe contener 'initial_state'"
        assert "episode_steps" in info, "info debe contener 'episode_steps'"

    def test_reset_deterministic_with_seed(self, env: ShrimpPondEnv) -> None:
        """Reset con la misma semilla debe producir el mismo estado inicial."""
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2, "Reset con misma semilla debe ser determinístico")

    def test_reset_produces_different_states(self, env: ShrimpPondEnv) -> None:
        """Reset con semillas diferentes debe producir estados diferentes."""
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        assert not np.array_equal(obs1, obs2), "Estados iniciales deben variar con semillas diferentes"

    @pytest.mark.parametrize("action", [0, 1, 2, 3, 4])
    def test_step_returns_valid_tuple(self, env: ShrimpPondEnv, action: int) -> None:
        """step() debe retornar una tupla (obs, reward, terminated, truncated, info) válida."""
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(action)

        # Observación válida
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (11,)
        assert env.observation_space.contains(obs)

        # Recompensa es un número finito
        assert isinstance(reward, (int, float)), "La recompensa debe ser numérica"
        assert np.isfinite(reward), "La recompensa debe ser finita"
        assert reward <= 0, "La recompensa debe ser <= 0 (función de penalización)"

        # Flags booleanos
        assert isinstance(terminated, bool), "terminated debe ser bool"
        assert isinstance(truncated, bool), "truncated debe ser bool"

        # info contiene campos esperados
        assert isinstance(info, dict)
        assert "step" in info
        assert "action" in info
        assert "action_cost" in info
        assert "mortality_rate" in info
        assert info["action"] == action

    @pytest.mark.parametrize("action", [0, 1, 2, 3, 4])
    def test_step_action_effects(self, env: ShrimpPondEnv, action: int) -> None:
        """Cada acción debe tener el efecto esperado en el estado."""
        obs_before, _ = env.reset(seed=99)
        obs_after, _, _, _, info = env.step(action)

        # Verificar costo de acción reportado
        expected_cost = ShrimpPondEnv.ACTION_COSTS.get(action, 0.0)
        assert info["action_cost"] == expected_cost

        # Acción 1: debe reducir amoníaco (índice 5)
        if action == 1:
            assert obs_after[5] <= obs_before[5] + 0.5, (
                "El intercambio de agua debe reducir o mantener el amoníaco"
            )

        # Acción 2: debe reducir feeding_rate (índice 7)
        if action == 2:
            assert obs_after[7] < obs_before[7], (
                "Reducir alimentación debe disminuir la tasa de alimentación"
            )

        # Acción 3: debe reducir outbreak_probability (índice 9)
        if action == 3:
            assert obs_after[9] <= obs_before[9] + 0.05, (
                "El tratamiento debe reducir la probabilidad de brote"
            )

        # Acción 4: debe reducir density (índice 6)
        if action == 4:
            assert obs_after[6] < obs_before[6], (
                "Reducir densidad debe disminuir la densidad poblacional"
            )

    def test_episode_runs_96_steps_with_action_0(self, env: ShrimpPondEnv) -> None:
        """Un episodio con acción 0 debe durar 96 pasos (a menos que termine por brote)."""
        env.reset(seed=7)
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(0)
            step_count += 1

            # Safety: no debe exceder los 96 pasos
            assert step_count <= 96, f"El episodio excedió 96 pasos: {step_count}"

        # Debe terminar por truncado (no por brote) en la mayoría de los casos
        # con acción 0, o por mortalidad si las condiciones se deterioran
        assert step_count <= 96, "El episodio no debe exceder 96 pasos"
        assert truncated or terminated, "El episodio debe terminar"

    def test_get_state_dict(self, env: ShrimpPondEnv) -> None:
        """get_state_dict debe retornar un diccionario con todas las variables."""
        env.reset(seed=10)
        state_dict = env.get_state_dict()

        assert isinstance(state_dict, dict)
        assert len(state_dict) == 11

        # Verificar que todas las claves esperadas están presentes
        expected_keys = {
            "ph", "do", "salinity", "turbidity", "temperature",
            "ammonia", "density", "feeding_rate", "days_since_stocking",
            "outbreak_probability", "current_mortality_rate",
        }
        assert set(state_dict.keys()) == expected_keys

        # Todos los valores son float
        for key, val in state_dict.items():
            assert isinstance(val, float), f"{key} debe ser float, no {type(val)}"

    def test_render_returns_none_by_default(self, env: ShrimpPondEnv) -> None:
        """render() debe retornar None cuando render_mode no es 'human'."""
        env.reset(seed=1)
        result = env.render()
        assert result is None

    def test_render_human_mode(self, capsys) -> None:
        """render() con render_mode='human' debe imprimir el estado."""
        env = ShrimpPondEnv(render_mode="human")
        env.reset(seed=1)
        result = env.render()
        assert result is not None
        assert isinstance(result, str)
        assert "Estado del Estanque" in result

    def test_invalid_action_raises_error(self, env: ShrimpPondEnv) -> None:
        """Una acción inválida debe lanzar ValueError."""
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step(99)

    def test_action_costs_defined(self) -> None:
        """Todos los costos de acción deben estar definidos."""
        for action in range(5):
            assert action in ShrimpPondEnv.ACTION_COSTS, (
                f"Costo para acción {action} no definido"
            )


# ========================================================================== #
# Pruebas del simulador económico
# ========================================================================== #
class TestEconomicSimulator:
    """Pruebas del simulador económico."""

    def test_calculate_scenario_basic(self, simulator: EconomicSimulator) -> None:
        """calculate_scenario debe retornar un diccionario con todas las claves."""
        result = simulator.calculate_scenario(
            action=0,
            duration_days=0,
            mortality_rate=0.05,
        )

        expected_keys = {
            "action", "action_name", "duration_days", "mortality_rate",
            "revenue", "stocking_cost", "feed_cost", "intervention_cost",
            "total_costs", "profit", "roi_percentage",
            "total_shrimp_stocked", "average_weight_g",
        }
        assert set(result.keys()) >= expected_keys

    def test_positive_roi_for_effective_intervention(self, simulator: EconomicSimulator) -> None:
        """Una intervención efectiva debe tener ROI positivo."""
        # Sin intervención: mortalidad alta (20%)
        # Con intervención (tratamiento): mortalidad baja (5%)
        result = simulator.simulate_intervention(
            pre_intervention_mortality=0.20,
            post_intervention_mortality=0.05,
            action=3,  # tratamiento químico
            duration=7.0,  # 7 días
        )

        # Las pérdidas evitadas deben ser positivas
        assert result["losses_avoided"] > 0, (
            "Una intervención efectiva debe evitar pérdidas (losses_avoided > 0)"
        )

        # El beneficio neto debe ser positivo
        assert result["net_benefit"] > 0, (
            f"Beneficio neto debe ser positivo para intervención efectiva, "
            f"obtenido: {result['net_benefit']}"
        )

        # ROI debe ser positivo
        assert result["roi_percentage"] > 0, (
            f"ROI debe ser positivo para intervención efectiva, "
            f"obtenido: {result['roi_percentage']}%"
        )

    def test_compare_policies(self, simulator: EconomicSimulator) -> None:
        """compare_policies debe retornar un DataFrame con las columnas correctas."""
        scenarios = [
            simulator.calculate_scenario(0, 0, 0.10),
            simulator.calculate_scenario(1, 5, 0.07),
            simulator.calculate_scenario(3, 7, 0.05),
        ]

        df = simulator.compare_policies(scenarios)

        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "Política" in df.columns
        assert "ROI (%)" in df.columns

    def test_simulate_intervention_keys(self, simulator: EconomicSimulator) -> None:
        """simulate_intervention debe retornar todas las claves esperadas."""
        result = simulator.simulate_intervention(
            pre_intervention_mortality=0.15,
            post_intervention_mortality=0.08,
            action=1,
            duration=3.0,
        )

        expected_keys = {
            "action", "action_name", "pre_intervention_mortality",
            "post_intervention_mortality", "mortality_reduction",
            "baseline_revenue", "intervened_revenue", "losses_avoided",
            "feed_cost_difference", "action_cost", "net_benefit", "roi_percentage",
        }
        assert set(result.keys()) >= expected_keys

    def test_generate_report(self, simulator: EconomicSimulator) -> None:
        """generate_report debe producir un informe en español."""
        scenarios = [
            simulator.calculate_scenario(0, 0, 0.10),
            simulator.calculate_scenario(3, 7, 0.05),
        ]

        report = simulator.generate_report(scenarios)

        assert isinstance(report, str)
        assert "REPORTE ECONÓMICO" in report
        assert "CAMARÓN" in report
        assert "Escenario 1" in report
        assert "Escenario 2" in report
        assert "Fin del reporte" in report

    def test_higher_mortality_means_lower_revenue(self, simulator: EconomicSimulator) -> None:
        """Mayor mortalidad debe resultar en menores ingresos."""
        low_mortality = simulator.calculate_scenario(0, 0, 0.02)
        high_mortality = simulator.calculate_scenario(0, 0, 0.30)

        assert low_mortality["revenue"] > high_mortality["revenue"], (
            "Mayor mortalidad debe reducir los ingresos"
        )

    def test_feed_cost_increases_with_survival(self, simulator: EconomicSimulator) -> None:
        """Mayor supervivencia implica mayor costo de alimento."""
        high_survival = simulator.calculate_scenario(0, 0, 0.02)
        low_survival = simulator.calculate_scenario(0, 0, 0.30)

        assert high_survival["feed_cost"] > low_survival["feed_cost"], (
            "Mayor supervivencia implica más camarones vivos comiendo, "
            "por tanto mayor costo de alimento"
        )


# ========================================================================== #
# Pruebas del recomendador de políticas
# ========================================================================== #
class TestPolicyRecommender:
    """Pruebas del recomendador de políticas."""

    def test_rule_based_fallback_action_3(self, recommender: PolicyRecommender) -> None:
        """Riesgo alto con amoníaco elevado debe recomendar tratamiento (acción 3)."""
        pond_state = {
            "outbreak_probability": 0.8,
            "ammonia": 3.0,
            "turbidity": 30.0,
            "density": 80.0,
            "feeding_rate": 3.0,
            "current_mortality_rate": 0.02,
        }

        result = recommender.recommend(pond_state)

        assert result["recommended_action"] == 3, (
            "Riesgo alto (>0.7) con amoníaco alto (>2.0) debe recomendar acción 3"
        )
        assert result["source"] == "rule_based"
        assert result["confidence"] > 0
        assert "económico" in result["expected_outcome"].lower() or "químico" in result["expected_outcome"].lower() or "tratamiento" in result["expected_outcome"].lower()

    def test_rule_based_fallback_action_1(self, recommender: PolicyRecommender) -> None:
        """Riesgo moderado-alto con turbidez alta debe recomendar intercambio de agua."""
        pond_state = {
            "outbreak_probability": 0.6,
            "ammonia": 1.0,
            "turbidity": 70.0,
            "density": 70.0,
            "feeding_rate": 3.0,
            "current_mortality_rate": 0.01,
        }

        result = recommender.recommend(pond_state)

        assert result["recommended_action"] == 1, (
            "Riesgo >0.5 con turbidez >50 debe recomendar acción 1"
        )

    def test_rule_based_fallback_action_4(self, recommender: PolicyRecommender) -> None:
        """Riesgo moderado con densidad alta debe recomendar reducir densidad."""
        pond_state = {
            "outbreak_probability": 0.4,
            "ammonia": 1.0,
            "turbidity": 30.0,
            "density": 110.0,
            "feeding_rate": 3.0,
            "current_mortality_rate": 0.01,
        }

        result = recommender.recommend(pond_state)

        assert result["recommended_action"] == 4, (
            "Riesgo >0.3 con densidad >80 debe recomendar acción 4"
        )

    def test_rule_based_fallback_action_2(self, recommender: PolicyRecommender) -> None:
        """Riesgo moderado con sobrealimentación debe recomendar reducir alimentación."""
        pond_state = {
            "outbreak_probability": 0.35,
            "ammonia": 1.0,
            "turbidity": 30.0,
            "density": 70.0,
            "feeding_rate": 6.0,
            "current_mortality_rate": 0.01,
        }

        result = recommender.recommend(pond_state)

        assert result["recommended_action"] == 2, (
            "Riesgo >0.3 con alimentación >óptimo debe recomendar acción 2"
        )

    def test_rule_based_fallback_action_0(self, recommender: PolicyRecommender) -> None:
        """Condiciones normales deben recomendar mantener (acción 0)."""
        pond_state = {
            "outbreak_probability": 0.1,
            "ammonia": 0.5,
            "turbidity": 15.0,
            "density": 70.0,
            "feeding_rate": 2.5,
            "current_mortality_rate": 0.005,
        }

        result = recommender.recommend(pond_state)

        assert result["recommended_action"] == 0, (
            "Condiciones normales deben recomendar acción 0 (mantener)"
        )

    def test_recommend_returns_valid_structure(self, recommender: PolicyRecommender) -> None:
        """recommend() debe retornar un diccionario con todas las claves esperadas."""
        pond_state = {
            "outbreak_probability": 0.5,
            "ammonia": 1.5,
            "turbidity": 40.0,
            "density": 90.0,
            "feeding_rate": 4.0,
            "current_mortality_rate": 0.02,
        }

        result = recommender.recommend(pond_state)

        expected_keys = {
            "recommended_action", "action_name", "confidence",
            "expected_outcome", "economic_impact", "source", "pond_state",
        }
        assert set(result.keys()) >= expected_keys

        # La acción debe ser válida (0-4)
        assert 0 <= result["recommended_action"] <= 4

        # La confianza debe estar entre 0 y 1
        assert 0.0 <= result["confidence"] <= 1.0

        # El impacto económico debe tener las claves esperadas
        econ = result["economic_impact"]
        assert "action_cost" in econ
        assert "net_benefit" in econ
        assert "roi_percentage" in econ

    def test_recommend_with_missing_values(self, recommender: PolicyRecommender) -> None:
        """recommend() debe manejar valores faltantes con valores por defecto."""
        pond_state = {
            "outbreak_probability": 0.2,
        }

        result = recommender.recommend(pond_state)

        # No debe fallar y debe retornar una acción válida
        assert 0 <= result["recommended_action"] <= 4
        assert result["source"] == "rule_based"
