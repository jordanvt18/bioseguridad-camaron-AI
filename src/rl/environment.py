"""
Entorno de aprendizaje por refuerzo para políticas de bioseguridad en camaronería.

Define un entorno personalizado de Gymnasium que simula un estanque camaronero,
modelando dinámicas de sensores, mortalidad y brotes de enfermedad.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple


class ShrimpPondEnv(gym.Env):
    """
    Entorno de simulación de estanque camaronero para RL.

    Espacio de observación (11,):
        - pH, oxígeno disuelto (DO), salinidad, turbidez, temperatura
        - amoníaco, densidad, tasa de alimentación
        - días desde siembra, probabilidad de brote, tasa de mortalidad actual

    Espacio de acciones (Discrete, 5):
        0: Sin acción (mantener)
        1: Aumentar intercambio de agua
        2: Ajustar alimentación (reducir 20%)
        3: Aplicar tratamiento químico
        4: Reducir densidad (transferir camarones)
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ #
    # Constantes del modelo
    # ------------------------------------------------------------------ #
    EPISODE_STEPS = 96  # 24 h a intervalos de 15 min
    MORTALITY_THRESHOLD = 0.15  # 15 % de mortalidad → brote/terminación

    # Costos por acción ($ por paso de 15 min)
    ACTION_COSTS: Dict[int, float] = {
        0: 0.0,
        1: 12.5,   # $50/día ÷ 96 pasos ≈ $0.52/paso, aproximado a 12.5 por día
        2: 0.0,
        3: 50.0,   # $200 por tratamiento
        4: 500.0,  # $500 logisticos
    }

    # Pesos de la función de recompensa
    ALPHA = 10.0   # peso mortalidad
    BETA = 0.01    # peso costo
    GAMMA = 5.0    # peso impacto ambiental

    # Límites de sensores (min, max)
    SENSOR_BOUNDS = {
        "ph": (6.5, 9.0),
        "do": (2.0, 9.0),        # oxígeno disuelto mg/L
        "salinity": (10.0, 40.0),  # ppt
        "turbidity": (0.0, 100.0),  # NTU
        "temperature": (24.0, 34.0),  # °C
        "ammonia": (0.0, 5.0),    # mg/L
        "density": (10.0, 150.0),  # camarones/m²
        "feeding_rate": (0.0, 10.0),  # kg/día
        "days_since_stocking": (0.0, 120.0),
        "outbreak_probability": (0.0, 1.0),
        "current_mortality_rate": (0.0, 1.0),
    }

    # Nombres de las variables de estado (orden = índice en el vector)
    STATE_NAMES = [
        "ph", "do", "salinity", "turbidity", "temperature",
        "ammonia", "density", "feeding_rate", "days_since_stocking",
        "outbreak_probability", "current_mortality_rate",
    ]

    # ------------------------------------------------------------------ #
    # Inicialización
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        alpha: float = 10.0,
        beta: float = 0.01,
        gamma: float = 5.0,
        max_steps: int = EPISODE_STEPS,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Inicializa el entorno.

        Args:
            alpha: Peso de la mortalidad en la recompensa.
            beta: Peso del costo de acción en la recompensa.
            gamma: Peso del impacto ambiental en la recompensa.
            max_steps: Número máximo de pasos por episodio.
            render_mode: Modo de renderizado ('human' o None).
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Espacio de observación: vector continuo de 11 dimensiones
        low = np.array([self.SENSOR_BOUNDS[n][0] for n in self.STATE_NAMES], dtype=np.float32)
        high = np.array([self.SENSOR_BOUNDS[n][1] for n in self.STATE_NAMES], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Espacio de acciones: discreto de 5 acciones
        self.action_space = spaces.Discrete(5)

        # Estado interno
        self._state: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Utilidades internas
    # ------------------------------------------------------------------ #
    def _random_initial_state(self) -> np.ndarray:
        """Genera un estado inicial aleatorio realista."""
        s = np.zeros(11, dtype=np.float32)

        s[0] = self._rng.uniform(7.2, 8.5)    # pH
        s[1] = self._rng.uniform(4.0, 7.0)    # DO
        s[2] = self._rng.uniform(20.0, 35.0)  # salinidad
        s[3] = self._rng.uniform(5.0, 40.0)   # turbidez
        s[4] = self._rng.uniform(26.0, 32.0)  # temperatura
        s[5] = self._rng.uniform(0.1, 1.5)    # amoníaco
        s[6] = self._rng.uniform(60.0, 120.0)  # densidad
        s[7] = self._rng.uniform(2.0, 6.0)    # feeding_rate
        s[8] = self._rng.uniform(1.0, 60.0)   # días desde siembra
        s[9] = self._rng.uniform(0.02, 0.20)   # prob. de brote inicial
        s[10] = self._rng.uniform(0.001, 0.03)  # mortalidad inicial

        return s

    def _clamp_state(self, s: np.ndarray) -> np.ndarray:
        """Mantiene los valores dentro de los límites definidos."""
        for i, name in enumerate(self.STATE_NAMES):
            lo, hi = self.SENSOR_BOUNDS[name]
            s[i] = np.clip(s[i], lo, hi)
        return s

    def _environmental_impact(self, state: np.ndarray) -> float:
        """
        Calcula un índice de impacto ambiental (0 = óptimo, 1 = crítico).

        Considera amoníaco, turbidez y oxígeno disuelto.
        """
        ammonia_norm = state[5] / self.SENSOR_BOUNDS["ammonia"][1]
        turbidity_norm = state[3] / self.SENSOR_BOUNDS["turbidity"][1]
        do_norm = 1.0 - (state[1] / self.SENSOR_BOUNDS["do"][1])  # bajo DO = alto impacto

        impact = (ammonia_norm + turbidity_norm + do_norm) / 3.0
        return float(np.clip(impact, 0.0, 1.0))

    def _update_dynamics(self, action: int) -> None:
        """
        Actualiza el estado interno según la acción tomada y dinámica natural.

        Args:
            action: Acción ejecutada (0-4).
        """
        s = self._state.copy()
        dt = 15.0 / 60.0 / 24.0  # fracción de día (15 min)

        # --- Deriva natural de sensores ---
        s[0] += self._rng.normal(0, 0.02)              # pH deriva leve
        s[1] += self._rng.normal(-0.05, 0.1)           # DO tiende a bajar
        s[2] += self._rng.normal(0, 0.1)               # salinidad estable
        s[3] += self._rng.normal(0.5, 1.0)             # turbidez tiende a subir
        s[4] += self._rng.normal(0, 0.1)               # temperatura fluctúa
        s[5] += self._rng.normal(0.02, 0.03)           # amoníaco sube naturalmente
        s[6] += self._rng.normal(0, 0.1)               # densidad estable
        s[7] += self._rng.normal(0, 0.05)              # alimentación fluctúa
        s[8] += dt                                      # avanza el tiempo
        s[10] += self._rng.normal(0.001, 0.005)        # mortalidad sube lentamente

        # --- Efectos de cada acción ---
        if action == 1:
            # Aumentar intercambio de agua
            s[5] -= self._rng.uniform(0.1, 0.3)        # reduce amoníaco
            s[3] -= self._rng.uniform(5.0, 15.0)       # reduce turbidez
            s[1] += self._rng.uniform(0.2, 0.5)        # aumenta DO
            s[2] += self._rng.uniform(-1.0, 1.0)       # salinidad puede cambiar

        elif action == 2:
            # Reducir alimentación 20%
            s[7] *= 0.8
            s[5] -= self._rng.uniform(0.02, 0.08)      # leve reducción de amoníaco

        elif action == 3:
            # Aplicar tratamiento químico
            s[9] -= self._rng.uniform(0.15, 0.35)      # reduce prob. de brote
            s[5] -= self._rng.uniform(0.05, 0.15)      # reduce amoníaco
            s[3] -= self._rng.uniform(2.0, 8.0)        # reduce turbidez

        elif action == 4:
            # Reducir densidad 10%
            s[6] *= 0.9
            s[5] -= self._rng.uniform(0.03, 0.10)      # menos densidad = menos amoníaco
            s[9] -= self._rng.uniform(0.05, 0.15)      # reduce prob. de brote

        # --- Actualizar probabilidad de brote según condiciones ---
        risk_factor = 0.0
        if s[5] > 2.0:       # amoníaco alto
            risk_factor += 0.02
        if s[3] > 50.0:      # turbidez alta
            risk_factor += 0.015
        if s[1] < 3.0:       # oxígeno bajo
            risk_factor += 0.02
        if s[4] > 32.0:      # temperatura alta
            risk_factor += 0.01
        if s[6] > 100.0:     # densidad alta
            risk_factor += 0.01

        s[9] += risk_factor
        s[9] = np.clip(s[9], 0.0, 1.0)

        # --- Actualizar mortalidad según condiciones ---
        if s[9] > 0.5:
            s[10] += self._rng.uniform(0.002, 0.01)
        if s[5] > 3.0:
            s[10] += self._rng.uniform(0.001, 0.005)
        if s[1] < 2.5:
            s[10] += self._rng.uniform(0.002, 0.008)

        s[10] = np.clip(s[10], 0.0, 1.0)

        self._state = self._clamp_state(s)

    def _compute_reward(self, action: int) -> float:
        """
        Calcula la recompensa: R = -(α·mortalidad + β·costo + γ·impacto).

        Args:
            action: Acción ejecutada.

        Returns:
            Recompensa escalar.
        """
        mortality = float(self._state[10])
        cost = self.ACTION_COSTS.get(action, 0.0)
        env_impact = self._environmental_impact(self._state)

        reward = -(
            self.alpha * mortality
            + self.beta * cost
            + self.gamma * env_impact
        )
        return float(reward)

    def _is_terminated(self) -> bool:
        """Verifica si el episodio debe terminar por brote."""
        return float(self._state[10]) >= self.MORTALITY_THRESHOLD

    # ------------------------------------------------------------------ #
    # API de Gymnasium
    # ------------------------------------------------------------------ #
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Ejecuta un paso del entorno.

        Args:
            action: Acción a ejecutar (0-4).

        Returns:
            Tuple (observation, reward, terminated, truncated, info).
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")

        self._update_dynamics(action)
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        self._step_count += 1
        truncated = self._step_count >= self.max_steps

        info: Dict[str, Any] = {
            "step": self._step_count,
            "action": action,
            "action_name": self._action_name(action),
            "action_cost": self.ACTION_COSTS.get(action, 0.0),
            "mortality_rate": float(self._state[10]),
            "outbreak_probability": float(self._state[9]),
            "environmental_impact": self._environmental_impact(self._state),
        }

        if terminated:
            info["termination_reason"] = "brote_detectado"
        elif truncated:
            info["termination_reason"] = "episodio_completado"

        return self._state.copy(), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reinicia el entorno con un estado inicial aleatorio.

        Args:
            seed: Semilla opcional para reproducibilidad.
            options: Opciones adicionales (no usadas actualmente).

        Returns:
            Tuple (observation, info).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._state = self._random_initial_state()
        self._step_count = 0

        info: Dict[str, Any] = {
            "initial_state": self.get_state_dict(),
            "episode_steps": self.max_steps,
        }
        return self._state.copy(), info

    def render(self) -> Optional[str]:
        """Renderiza el estado actual del entorno como texto."""
        if self.render_mode != "human":
            return None

        lines = [
            f"=== Estado del Estanque (Paso {self._step_count}/{self.max_steps}) ===",
        ]
        for i, name in enumerate(self.STATE_NAMES):
            lines.append(f"  {name:25s}: {self._state[i]:.4f}")
        lines.append(f"  Impacto ambiental: {self._environmental_impact(self._state):.4f}")
        lines.append("=" * 55)
        text = "\n".join(lines)
        print(text)
        return text

    def get_state_dict(self) -> Dict[str, float]:
        """Retorna el estado actual como diccionario {nombre: valor}."""
        if self._state is None:
            return {}
        return {name: float(val) for name, val in zip(self.STATE_NAMES, self._state)}

    @staticmethod
    def _action_name(action: int) -> str:
        """Nombre legible de cada acción."""
        names = {
            0: "Mantener",
            1: "Intercambio de agua",
            2: "Reducir alimentación",
            3: "Tratamiento químico",
            4: "Reducir densidad",
        }
        return names.get(action, "Desconocida")
