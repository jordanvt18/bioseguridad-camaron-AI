"""
Recomendador de políticas de intervención.

Combina el modelo de aprendizaje por refuerzo (RL) entrenado y el
predictor de brotes de aprendizaje profundo (DL) para recomendar
la acción óptima dado el estado del estanque.

Incluye un sistema de respaldo basado en reglas expertas cuando
los modelos entrenados no están disponibles.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .environment import ShrimpPondEnv
from .economic_simulator import EconomicSimulator


class PolicyRecommender:
    """
    Recomendador de políticas de bioseguridad para estanques camaroneros.

    Usa un modelo RL entrenado cuando está disponible; de lo contrario,
    aplica reglas heurísticas basadas en conocimiento experto.
    """

    # Nombres legibles de acciones
    ACTION_NAMES: Dict[int, str] = {
        0: "Mantener (sin acción)",
        1: "Aumentar intercambio de agua",
        2: "Reducir alimentación (20%)",
        3: "Aplicar tratamiento químico",
        4: "Reducir densidad (transferir camarones)",
    }

    # Descripciones de resultados esperados por acción
    ACTION_OUTCOMES: Dict[int, str] = {
        0: "Mantener condiciones actuales. Monitoreo continuo recomendado.",
        1: "Reducción de amoníaco y turbidez. Mejora en oxígeno disuelto.",
        2: "Reducción de costo de alimentación y leve mejora en calidad de agua.",
        3: "Reducción significativa de probabilidad de brote. Costo elevado.",
        4: "Reducción de densidad poblacional. Disminuye estrés y riesgo de brote.",
    }

    def __init__(
        self,
        rl_model_path: Optional[str] = None,
        dl_model_path: Optional[str] = None,
        economic_simulator: Optional[EconomicSimulator] = None,
    ) -> None:
        """
        Inicializa el recomendador.

        Args:
            rl_model_path: Ruta al modelo RL entrenado (.zip de SB3).
            dl_model_path: Ruta al modelo DL predictor de brotes (.h5/.pt).
            economic_simulator: Instancia del simulador económico. Si es None,
                                se crea una con parámetros por defecto.
        """
        self.rl_model = None
        self.dl_model = None
        self.economic_simulator = economic_simulator or EconomicSimulator()
        self.env = ShrimpPondEnv()

        # Intentar cargar modelo RL
        if rl_model_path:
            try:
                from stable_baselines3 import PPO
                self.rl_model = PPO.load(rl_model_path)
            except Exception:
                try:
                    from stable_baselines3 import DQN
                    self.rl_model = DQN.load(rl_model_path)
                except Exception:
                    self.rl_model = None

        # Intentar cargar modelo DL
        if dl_model_path:
            try:
                import tensorflow as tf
                self.dl_model = tf.keras.models.load_model(dl_model_path)
            except Exception:
                try:
                    import torch
                    self.dl_model = torch.load(dl_model_path, map_location="cpu")
                    self.dl_model.eval()
                except Exception:
                    self.dl_model = None

    # ------------------------------------------------------------------ #
    # Recomendación basada en reglas (respaldo)
    # ------------------------------------------------------------------ #
    def _rule_based_recommendation(
        self,
        outbreak_prob: float,
        ammonia: float,
        turbidity: float,
        density: float,
        feeding_rate: float,
    ) -> Tuple[int, float, str]:
        """
        Sistema de reglas expertas para recomendar una acción.

        Reglas (en orden de prioridad):
            1. outbreak_prob > 0.7 y amoníaco > 2.0 → acción 3 (tratamiento)
            2. outbreak_prob > 0.5 y turbidez > 50  → acción 1 (intercambio agua)
            3. outbreak_prob > 0.3 y densidad > 80   → acción 4 (reducir densidad)
            4. outbreak_prob > 0.3 y alimentación > óptimo → acción 2 (reducir alimentación)
            5. Else → acción 0 (mantener)

        Args:
            outbreak_prob: Probabilidad de brote (0-1).
            ammonia: Nivel de amoníaco (mg/L).
            turbidity: Turbidez (NTU).
            density: Densidad (camarones/m²).
            feeding_rate: Tasa de alimentación (kg/día).

        Returns:
            Tuple (action, confidence, reason).
        """
        # Alimentación óptima aproximada (3% del peso corporal promedio)
        # Para 100/m² en 1 ha: ~10000*100 camarones, ~24g promedio
        optimal_feed = 3.0  # kg/día aproximado como umbral

        if outbreak_prob > 0.7 and ammonia > 2.0:
            confidence = min(0.95, 0.7 + (outbreak_prob - 0.7) * 0.8 + (ammonia - 2.0) * 0.05)
            return 3, float(confidence), (
                "Riesgo alto de brote con amoníaco elevado. "
                "Se recomienda tratamiento químico inmediato."
            )

        if outbreak_prob > 0.5 and turbidity > 50:
            confidence = min(0.90, 0.5 + (outbreak_prob - 0.5) * 0.6 + (turbidity - 50) * 0.002)
            return 1, float(confidence), (
                "Riesgo moderado-alto de brote con turbidez elevada. "
                "Se recomienda aumentar el intercambio de agua."
            )

        if outbreak_prob > 0.3 and density > 80:
            confidence = min(0.85, 0.3 + (outbreak_prob - 0.3) * 0.5 + (density - 80) * 0.002)
            return 4, float(confidence), (
                "Riesgo moderado con alta densidad poblacional. "
                "Se recomienda reducir densidad transfiriendo camarones."
            )

        if outbreak_prob > 0.3 and feeding_rate > optimal_feed:
            confidence = min(0.80, 0.3 + (outbreak_prob - 0.3) * 0.5 + (feeding_rate - optimal_feed) * 0.03)
            return 2, float(confidence), (
                "Riesgo moderado con sobrealimentación detectada. "
                "Se recomienda reducir la alimentación en 20%."
            )

        # Sin acción necesaria
        confidence = max(0.5, 1.0 - outbreak_prob)
        return 0, float(confidence), (
            "Condiciones dentro de parámetros aceptables. "
            "Mantener monitoreo regular."
        )

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def recommend(self, pond_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recomienda la acción óptima dado el estado del estanque.

        Args:
            pond_state: Diccionario con al menos las siguientes claves:
                - outbreak_probability (float, 0-1)
                - ammonia (float, mg/L)
                - turbidity (float, NTU)
                - density (float, camarones/m²)
                - feeding_rate (float, kg/día)
                Opcionalmente puede incluir todas las variables del entorno.

        Returns:
            Diccionario con:
                - recommended_action: int (0-4)
                - action_name: str
                - confidence: float (0-1)
                - expected_outcome: str
                - economic_impact: dict
                - source: 'rl_model' | 'rule_based'
        """
        # Extraer variables relevantes con valores por defecto
        outbreak_prob = float(pond_state.get("outbreak_probability", 0.0))
        ammonia = float(pond_state.get("ammonia", 0.0))
        turbidity = float(pond_state.get("turbidity", 0.0))
        density = float(pond_state.get("density", 100.0))
        feeding_rate = float(pond_state.get("feeding_rate", 3.0))
        mortality_rate = float(pond_state.get("current_mortality_rate", 0.0))

        # Intentar usar el modelo RL entrenado
        if self.rl_model is not None:
            try:
                # Construir vector de observación
                state_vector = np.array([
                    pond_state.get("ph", 7.5),
                    pond_state.get("do", 5.0),
                    pond_state.get("salinity", 25.0),
                    turbidity,
                    pond_state.get("temperature", 28.0),
                    ammonia,
                    density,
                    feeding_rate,
                    pond_state.get("days_since_stocking", 30.0),
                    outbreak_prob,
                    mortality_rate,
                ], dtype=np.float32)

                action, _ = self.rl_model.predict(state_vector, deterministic=True)
                action = int(action)

                # Calcular confianza basada en la entropía de la política
                confidence = 0.85  # valor por defecto; SB3 no expone entropía fácilmente

                source = "rl_model"
                reason = self.ACTION_OUTCOMES.get(action, "Acción recomendada por modelo RL.")

            except Exception:
                # Fallback a reglas
                action, confidence, reason = self._rule_based_recommendation(
                    outbreak_prob, ammonia, turbidity, density, feeding_rate
                )
                source = "rule_based"
        else:
            # Sin modelo RL, usar reglas
            action, confidence, reason = self._rule_based_recommendation(
                outbreak_prob, ammonia, turbidity, density, feeding_rate
            )
            source = "rule_based"

        # Calcular impacto económico
        economic_impact = self._calculate_economic_impact(
            action, outbreak_prob, mortality_rate
        )

        return {
            "recommended_action": action,
            "action_name": self.ACTION_NAMES.get(action, "Desconocida"),
            "confidence": round(confidence, 4),
            "expected_outcome": reason,
            "economic_impact": economic_impact,
            "source": source,
            "pond_state": {
                "outbreak_probability": outbreak_prob,
                "ammonia": ammonia,
                "turbidity": turbidity,
                "density": density,
                "feeding_rate": feeding_rate,
                "current_mortality_rate": mortality_rate,
            },
        }

    def _calculate_economic_impact(
        self,
        action: int,
        outbreak_prob: float,
        current_mortality: float,
    ) -> Dict[str, Any]:
        """
        Calcula el impacto económico estimado de la acción recomendada.

        Args:
            action: Acción recomendada (0-4).
            outbreak_prob: Probabilidad de brote actual.
            current_mortality: Tasa de mortalidad actual.

        Returns:
            Diccionario con costo de acción, beneficio estimado y ROI.
        """
        # Mortalidad esperada sin intervención
        pre_mortality = min(0.5, current_mortality + outbreak_prob * 0.3)

        # Mortalidad esperada con intervención (estimación)
        reduction_map = {0: 0.0, 1: 0.05, 2: 0.02, 3: 0.15, 4: 0.10}
        reduction = reduction_map.get(action, 0.0)
        post_mortality = max(0.0, pre_mortality - reduction)

        # Usar el simulador económico
        impact = self.economic_simulator.simulate_intervention(
            pre_intervention_mortality=pre_mortality,
            post_intervention_mortality=post_mortality,
            action=action,
            duration=1.0,  # 1 día de intervención
        )

        return {
            "action_cost": impact["action_cost"],
            "losses_avoided": impact["losses_avoided"],
            "net_benefit": impact["net_benefit"],
            "roi_percentage": impact["roi_percentage"],
            "estimated_mortality_without_action": round(pre_mortality, 4),
            "estimated_mortality_with_action": round(post_mortality, 4),
        }
