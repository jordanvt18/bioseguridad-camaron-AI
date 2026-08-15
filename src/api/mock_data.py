"""
Generador y gestor de datos simulados para la API de Bioseguridad Inteligente.

Proporciona datos realistas de estanques camaroneros sin necesidad de modelos
entrenados de DL/RL. Genera 3 granjas con 15 estanques en total y simula
variaciones temporales en las lecturas de sensores.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from .schemas import (
    AlertLevel,
    ContributingFactor,
    PondStatus,
    RiskAssessment,
    RiskLevel,
    SensorReading,
)


# ---------------------------------------------------------------------------
# Rangos óptimos de parámetros de agua para camarón (Litopenaeus vannamei)
# ---------------------------------------------------------------------------

OPTIMAL_RANGES = {
    "ph": (7.5, 8.5),
    "dissolved_oxygen": (5.0, 8.0),
    "salinity": (15.0, 25.0),
    "turbidity": (10.0, 30.0),
    "temperature": (26.0, 30.0),
    "ammonia": (0.0, 0.5),
}

# Descripción humana de los rangos óptimos
OPTIMAL_RANGE_LABELS = {
    "ph": "7.5 – 8.5",
    "dissolved_oxygen": "5.0 – 8.0 mg/L",
    "salinity": "15 – 25 ppt",
    "turbidity": "10 – 30 NTU",
    "temperature": "26 – 30 °C",
    "ammonia": "0 – 0.5 mg/L",
}

# Nombres legibles de factores
FACTOR_LABELS = {
    "ph": "pH",
    "dissolved_oxygen": "Oxígeno disuelto",
    "salinity": "Salinidad",
    "turbidity": "Turbidez",
    "temperature": "Temperatura",
    "ammonia": "Amoníaco",
}

# Nombres de acciones de manejo
ACTION_NAMES = {
    0: "Mantener condiciones actuales",
    1: "Recambio de agua",
    2: "Reducir densidad de siembra",
    3: "Aplicar probióticos",
    4: "Cosecha parcial",
}


class MockDataManager:
    """
    Gestor de datos simulados para estanques camaroneros.

    Mantiene una base de datos en memoria de 3 granjas con 15 estanques.
    Cada estanque tiene ubicación, lecturas de sensores (con variación temporal),
    historial de brotes y densidad actual.
    """

    def __init__(self) -> None:
        """Inicializa la base de datos mock de estanques."""
        self._farms: dict[str, dict] = {}
        self._ponds: dict[str, dict] = {}
        self._rng = random.Random(42)
        self._np_rng = np.random.default_rng(seed=42)
        self._initialize_database()

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def _initialize_database(self) -> None:
        """Crea 3 granjas con 5 estanques cada una (15 total)."""
        farm_configs = [
            ("farm-001", "Granja Camarones del Pacífico", "Guayas, Ecuador"),
            ("farm-002", "Granja Acuícola El Oro", "El Oro, Ecuador"),
            ("farm-003", "Granja Mar Tropical", "Manabí, Ecuador"),
        ]

        pond_names_pool = [
            "Estanque A-1", "Estanque A-2", "Estanque B-1",
            "Estanque B-2", "Estanque C-1", "Estanque C-2",
            "Estanque D-1", "Estanque D-2", "Estanque E-1",
        ]

        pond_counter = 0
        for farm_id, farm_name, location in farm_configs:
            self._farms[farm_id] = {
                "name": farm_name,
                "location": location,
            }
            # 5 estanques por granja
            for i in range(5):
                pond_counter += 1
                pond_id = f"pond-{pond_counter:03d}"
                self._ponds[pond_id] = {
                    "pond_id": pond_id,
                    "farm_id": farm_id,
                    "name": pond_names_pool[i],
                    "location": location,
                    "density": self._rng.randint(30, 60),  # organismos/m²
                    "area_hectares": round(self._rng.uniform(2.0, 8.0), 1),
                    "days_in_cycle": self._rng.randint(15, 90),
                    "outbreak_history": self._generate_outbreak_history(),
                    "base_sensors": self._generate_base_sensors(),
                }

    def _generate_base_sensors(self) -> dict[str, float]:
        """Genera lecturas base de sensores con pequeña desviación del óptimo."""
        sensors = {}
        for param, (low, high) in OPTIMAL_RANGES.items():
            mid = (low + high) / 2
            spread = (high - low) / 2
            # 70% dentro del rango óptimo, 30% fuera
            if self._rng.random() < 0.7:
                sensors[param] = round(self._rng.uniform(low, high), 2)
            else:
                sensors[param] = round(
                    mid + self._rng.uniform(-spread * 2, spread * 2), 2
                )
        return sensors

    def _generate_outbreak_history(self) -> list[dict]:
        """Genera historial de brotes anteriores (0-3 eventos)."""
        n_events = self._rng.randint(0, 3)
        history = []
        for _ in range(n_events):
            days_ago = self._rng.randint(30, 365)
            event_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
            history.append({
                "date": event_date.isoformat(),
                "severity": round(self._rng.uniform(0.1, 0.9), 2),
                "mortality_rate": round(self._rng.uniform(0.05, 0.3), 2),
            })
        return history

    # ------------------------------------------------------------------
    # Lecturas de sensores con variación temporal
    # ------------------------------------------------------------------

    def _get_current_sensors(self, pond_id: str) -> dict[str, float]:
        """Obtiene lecturas actuales con pequeña variación temporal."""
        pond = self._ponds[pond_id]
        base = pond["base_sensors"]
        sensors = {}
        for param, value in base.items():
            # Variación gaussiana pequeña
            noise = self._np_rng.normal(0, abs(value) * 0.03)
            sensors[param] = round(max(0, value + noise), 2)
        return sensors

    def _build_sensor_reading(self, pond_id: str) -> SensorReading:
        """Construye un objeto SensorReading para un estanque."""
        sensors = self._get_current_sensors(pond_id)
        now = datetime.now(timezone.utc)
        return SensorReading(
            ph=sensors["ph"],
            dissolved_oxygen=sensors["dissolved_oxygen"],
            salinity=sensors["salinity"],
            turbidity=sensors["turbidity"],
            temperature=sensors["temperature"],
            ammonia=sensors["ammonia"],
            timestamp=now,
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def get_pond_status(self, pond_id: str) -> Optional[PondStatus]:
        """
        Obtiene el estado actual de un estanque.

        Args:
            pond_id: Identificador del estanque.

        Returns:
            PondStatus si el estanque existe, None si no.
        """
        pond = self._ponds.get(pond_id)
        if pond is None:
            return None

        sensors = self._build_sensor_reading(pond_id)
        alert_level = self._compute_alert_level(sensors)
        now = datetime.now(timezone.utc)

        return PondStatus(
            pond_id=pond_id,
            farm_id=pond["farm_id"],
            name=pond["name"],
            sensors=sensors,
            alert_level=alert_level,
            last_update=now,
        )

    def get_all_ponds(self) -> list[PondStatus]:
        """
        Obtiene el estado de todos los estanques.

        Returns:
            Lista de PondStatus para los 15 estanques.
        """
        return [
            self.get_pond_status(pid) for pid in sorted(self._ponds.keys())
        ]

    def get_farm_ponds(self, farm_id: str) -> list[str]:
        """Obtiene los IDs de estanques de una granja."""
        return [
            pid for pid, p in self._ponds.items() if p["farm_id"] == farm_id
        ]

    def get_farm_name(self, farm_id: str) -> Optional[str]:
        """Obtiene el nombre de una granja."""
        farm = self._farms.get(farm_id)
        return farm["name"] if farm else None

    def farm_exists(self, farm_id: str) -> bool:
        """Verifica si una granja existe."""
        return farm_id in self._farms

    def pond_exists(self, pond_id: str) -> bool:
        """Verifica si un estanque existe."""
        return pond_id in self._ponds

    # ------------------------------------------------------------------
    # Cálculo de riesgo
    # ------------------------------------------------------------------

    def _compute_alert_level(self, sensors: SensorReading) -> AlertLevel:
        """Calcula el nivel de alerta basado en lecturas de sensores."""
        risk = self._compute_risk_score(sensors)
        if risk >= 0.7:
            return AlertLevel.EMERGENCIA
        elif risk >= 0.5:
            return AlertLevel.ALERTA
        elif risk >= 0.3:
            return AlertLevel.PRECAUCION
        else:
            return AlertLevel.NORMAL

    def _compute_risk_score(self, sensors: SensorReading) -> float:
        """
        Calcula un puntaje de riesgo (0-1) usando reglas de umbral.

        Combina la severidad de desviación de cada parámetro respecto
        a su rango óptimo, ponderando amoníaco y oxígeno más fuertemente.
        """
        values = {
            "ph": sensors.ph,
            "dissolved_oxygen": sensors.dissolved_oxygen,
            "salinity": sensors.salinity,
            "turbidity": sensors.turbidity,
            "temperature": sensors.temperature,
            "ammonia": sensors.ammonia,
        }

        weights = {
            "ph": 0.15,
            "dissolved_oxygen": 0.25,
            "salinity": 0.10,
            "turbidity": 0.10,
            "temperature": 0.15,
            "ammonia": 0.25,
        }

        total_risk = 0.0
        for param, value in values.items():
            low, high = OPTIMAL_RANGES[param]
            mid = (low + high) / 2
            spread = (high - low) / 2

            if low <= value <= high:
                deviation = 0.0
            else:
                deviation = min(1.0, abs(value - mid) / (spread * 2))

            total_risk += weights[param] * deviation

        # Factor adicional por densidad alta (simulado)
        return round(min(1.0, total_risk), 4)

    def _compute_contributing_factors(
        self, sensors: SensorReading
    ) -> list[ContributingFactor]:
        """Calcula los factores contribuyentes al riesgo."""
        values = {
            "ph": sensors.ph,
            "dissolved_oxygen": sensors.dissolved_oxygen,
            "salinity": sensors.salinity,
            "turbidity": sensors.turbidity,
            "temperature": sensors.temperature,
            "ammonia": sensors.ammonia,
        }

        factors = []
        for param, value in values.items():
            low, high = OPTIMAL_RANGES[param]
            mid = (low + high) / 2
            spread = (high - low) / 2

            if low <= value <= high:
                severity = 0.0
            else:
                severity = round(
                    min(1.0, abs(value - mid) / (spread * 2)), 4
                )

            factors.append(ContributingFactor(
                factor=FACTOR_LABELS[param],
                value=value,
                optimal_range=OPTIMAL_RANGE_LABELS[param],
                severity=severity,
            ))

        # Ordenar por severidad descendente
        factors.sort(key=lambda f: f.severity, reverse=True)
        return factors

    def _risk_level_from_score(self, score: float) -> RiskLevel:
        """Convierte un puntaje de riesgo a nivel categórico."""
        if score >= 0.7:
            return RiskLevel.CRITICO
        elif score >= 0.5:
            return RiskLevel.ALTO
        elif score >= 0.3:
            return RiskLevel.MEDIO
        else:
            return RiskLevel.BAJO

    def simulate_outbreak_probability(self, pond_id: str) -> Optional[RiskAssessment]:
        """
        Simula la probabilidad de brote para un estanque.

        Usa reglas de umbral sobre las lecturas de sensores para generar
        una probabilidad realista. Incluye factores contribuyentes.

        Args:
            pond_id: Identificador del estanque.

        Returns:
            RiskAssessment si el estanque existe, None si no.
        """
        pond = self._ponds.get(pond_id)
        if pond is None:
            return None

        sensors = self._build_sensor_reading(pond_id)
        risk_score = self._compute_risk_score(sensors)

        # Ajustar por densidad y días de ciclo
        density_factor = min(0.2, (pond["density"] - 30) / 150)
        cycle_factor = min(0.15, pond["days_in_cycle"] / 600)
        outbreak_history_factor = min(0.15, len(pond["outbreak_history"]) * 0.05)

        probability = round(
            min(1.0, risk_score + density_factor + cycle_factor + outbreak_history_factor),
            4
        )

        risk_level = self._risk_level_from_score(probability)
        factors = self._compute_contributing_factors(sensors)

        # Añadir factores no sensoriales
        if density_factor > 0.05:
            factors.append(ContributingFactor(
                factor="Densidad de siembra",
                value=float(pond["density"]),
                optimal_range="30 – 45 organismos/m²",
                severity=round(min(1.0, density_factor * 5), 4),
            ))
        if outbreak_history_factor > 0.05:
            factors.append(ContributingFactor(
                factor="Historial de brotes",
                value=float(len(pond["outbreak_history"])),
                optimal_range="0 eventos",
                severity=round(min(1.0, outbreak_history_factor * 5), 4),
            ))

        factors.sort(key=lambda f: f.severity, reverse=True)

        return RiskAssessment(
            pond_id=pond_id,
            outbreak_probability=probability,
            risk_level=risk_level,
            contributing_factors=factors,
            timestamp=datetime.now(timezone.utc),
        )

    def get_farm_risks(self, farm_id: str) -> Optional[list[RiskAssessment]]:
        """
        Obtiene evaluaciones de riesgo para todos los estanques de una granja.

        Args:
            farm_id: Identificador de la granja.

        Returns:
            Lista de RiskAssessment, o None si la granja no existe.
        """
        if farm_id not in self._farms:
            return None

        pond_ids = self.get_farm_ponds(farm_id)
        results = []
        for pid in pond_ids:
            risk = self.simulate_outbreak_probability(pid)
            if risk:
                results.append(risk)
        return results

    # ------------------------------------------------------------------
    # Simulación económica
    # ------------------------------------------------------------------

    def simulate_action(
        self, pond_id: str, action: int, duration_days: int
    ) -> Optional[dict]:
        """
        Simula el resultado económico de una acción de manejo.

        Args:
            pond_id: Identificador del estanque.
            action: Acción a simular (0-4).
            duration_days: Duración en días.

        Returns:
            Diccionario con resultados económicos, o None si el estanque no existe.
        """
        pond = self._ponds.get(pond_id)
        if pond is None:
            return None

        risk = self.simulate_outbreak_probability(pond_id)
        if risk is None:
            return None

        base_prob = risk.outbreak_probability
        area = pond["area_hectares"]
        density = pond["density"]

        # Producción estimada (kg) sin intervención
        avg_weight = 12  # gramos promedio
        expected_production = area * 10000 * density * avg_weight / 1000  # kg
        price_per_kg = 6.50  # USD/kg

        # Costos base
        cost_per_hectare = 8000  # USD/ha/ciclo
        base_costs = area * cost_per_hectare * (duration_days / 120)

        # Ingresos base sin acción
        base_revenue = expected_production * price_per_kg * (duration_days / 120)
        base_losses = base_revenue * base_prob * 0.4  # 40% de pérdida si hay brote

        # Efecto de cada acción
        action_effects = {
            0: {"prob_reduction": 0.0, "cost": 0, "revenue_factor": 1.0},
            1: {"prob_reduction": 0.15, "cost": 500 * area, "revenue_factor": 0.98},
            2: {"prob_reduction": 0.25, "cost": 200 * area, "revenue_factor": 0.85},
            3: {"prob_reduction": 0.20, "cost": 1200 * area, "revenue_factor": 0.97},
            4: {"prob_reduction": 0.35, "cost": 300 * area, "revenue_factor": 0.70},
        }

        effect = action_effects[action]
        new_prob = max(0.0, base_prob - effect["prob_reduction"])
        action_revenue = base_revenue * effect["revenue_factor"]
        action_costs = base_costs + effect["cost"]
        losses_avoided = base_losses * (effect["prob_reduction"] / max(base_prob, 0.01))

        profit = action_revenue - action_costs
        roi = (profit / action_costs * 100) if action_costs > 0 else 0.0
        net_benefit = (losses_avoided - effect["cost"]) if action > 0 else 0.0

        return {
            "action_taken": action,
            "action_name": ACTION_NAMES[action],
            "duration_days": duration_days,
            "revenue": round(action_revenue, 2),
            "costs": round(action_costs, 2),
            "profit": round(profit, 2),
            "roi_percentage": round(roi, 2),
            "losses_avoided": round(losses_avoided, 2),
            "net_benefit": round(net_benefit, 2),
        }

    # ------------------------------------------------------------------
    # Recomendación de política
    # ------------------------------------------------------------------

    def get_policy_recommendation(self, pond_id: str) -> Optional[dict]:
        """
        Genera una recomendación de política basada en el riesgo actual.

        Usa heurísticas simples: si el riesgo es alto, recomienda acciones
        más agresivas; si es bajo, recomienda mantener.

        Args:
            pond_id: Identificador del estanque.

        Returns:
            Diccionario con recomendación, o None si el estanque no existe.
        """
        risk = self.simulate_outbreak_probability(pond_id)
        if risk is None:
            return None

        prob = risk.outbreak_probability

        # Heurística de recomendación
        if prob < 0.2:
            recommended = 0  # Mantener
            confidence = 0.85
            outcome = (
                "Las condiciones del estanque son estables. "
                "Se recomienda monitoreo rutinario y mantenimiento de parámetros."
            )
        elif prob < 0.4:
            recommended = 3  # Probióticos
            confidence = 0.72
            outcome = (
                "Riesgo moderado detectado. La aplicación de probióticos "
                "puede mejorar la calidad del agua y reducir el riesgo de brote."
            )
        elif prob < 0.6:
            recommended = 1  # Recambio de agua
            confidence = 0.68
            outcome = (
                "Riesgo elevado. Se recomienda recambio de agua urgente "
                "para restablecer parámetros óptimos de calidad."
            )
        elif prob < 0.8:
            recommended = 2  # Reducir densidad
            confidence = 0.65
            outcome = (
                "Riesgo alto. Se recomienda reducir la densidad de siembra "
                "para disminuir el estrés y la transmisión de patógenos."
            )
        else:
            recommended = 4  # Cosecha parcial
            confidence = 0.78
            outcome = (
                "Riesgo crítico. Se recomienda cosecha parcial inmediata "
                "para mitigar pérdidas económicas por potencial brote."
            )

        # Calcular impacto económico de la recomendación
        sim = self.simulate_action(pond_id, recommended, 30)
        economic_impact = sim["net_benefit"] if sim else 0.0

        return {
            "pond_id": pond_id,
            "recommended_action": recommended,
            "action_name": ACTION_NAMES[recommended],
            "confidence": confidence,
            "expected_outcome": outcome,
            "economic_impact": round(economic_impact, 2),
        }

    @property
    def n_ponds(self) -> int:
        """Número total de estanques."""
        return len(self._ponds)

    @property
    def farms(self) -> dict[str, dict]:
        """Diccionario de granjas."""
        return self._farms
