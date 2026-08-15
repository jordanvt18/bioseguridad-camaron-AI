"""
Simulador económico para estanques camaroneros.

Modela costos, ingresos y retorno de inversión (ROI) de diferentes
políticas de intervención de bioseguridad.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class EconomicSimulator:
    """
    Simulador económico de cultivo de camarón.

    Modela:
        - Costo de siembra: $0.05 por post-larva, densidad 100/m²
        - Costo de alimento: $1.50/kg, alimentación diaria = 3% peso corporal
        - Precio de venta: $4.00/kg camarón adulto
        - Ciclo de cultivo: 120 días
        - Ganancia de peso promedio: 0.2 g/día
    """

    # ------------------------------------------------------------------ #
    # Parámetros económicos
    # ------------------------------------------------------------------ #
    POST_LARVA_COST = 0.05        # $ por post-larva
    DEFAULT_DENSITY = 100.0        # camarones/m²
    FEED_COST_PER_KG = 1.50        # $/kg de alimento
    FEED_RATIO = 0.03              # 3% del peso corporal por día
    SALE_PRICE_PER_KG = 4.00       # $/kg de camarón adulto
    CULTURE_CYCLE_DAYS = 120       # días por ciclo
    WEIGHT_GAIN_PER_DAY = 0.2      # g/día
    INITIAL_WEIGHT = 0.01          # g (post-larva)
    POND_AREA = 10000.0            # m² (1 hectárea)

    # Costos de acción por día ($)
    ACTION_DAILY_COSTS: Dict[int, float] = {
        0: 0.0,
        1: 50.0,     # intercambio de agua
        2: 0.0,      # reducir alimentación (ahorra costo)
        3: 200.0,    # tratamiento químico
        4: 500.0,    # reducción de densidad (logística)
    }

    ACTION_NAMES: Dict[int, str] = {
        0: "Mantener",
        1: "Intercambio de agua",
        2: "Reducir alimentación",
        3: "Tratamiento químico",
        4: "Reducir densidad",
    }

    def __init__(
        self,
        pond_area: float = POND_AREA,
        density: float = DEFAULT_DENSITY,
        sale_price: float = SALE_PRICE_PER_KG,
        feed_cost: float = FEED_COST_PER_KG,
        post_larva_cost: float = POST_LARVA_COST,
        cycle_days: int = CULTURE_CYCLE_DAYS,
        weight_gain: float = WEIGHT_GAIN_PER_DAY,
    ) -> None:
        """
        Inicializa el simulador económico.

        Args:
            pond_area: Área del estanque en m².
            density: Densidad de siembra en camarones/m².
            sale_price: Precio de venta por kg de camarón ($/kg).
            feed_cost: Costo del alimento por kg ($/kg).
            post_larva_cost: Costo por post-larva ($).
            cycle_days: Duración del ciclo de cultivo (días).
            weight_gain: Ganancia de peso diaria (g/día).
        """
        self.pond_area = pond_area
        self.density = density
        self.sale_price = sale_price
        self.feed_cost = feed_cost
        self.post_larva_cost = post_larva_cost
        self.cycle_days = cycle_days
        self.weight_gain = weight_gain

    # ------------------------------------------------------------------ #
    # Cálculos base
    # ------------------------------------------------------------------ #
    def _total_shrimp_stocked(self) -> int:
        """Calcula el número total de camarones sembrados."""
        return int(self.pond_area * self.density)

    def _average_adult_weight(self, days: Optional[int] = None) -> float:
        """
        Peso promedio del camarón después de `days` días (en gramos).

        Args:
            days: Días de cultivo. Por defecto, el ciclo completo.

        Returns:
            Peso promedio en gramos.
        """
        days = days or self.cycle_days
        return self.INITIAL_WEIGHT + self.weight_gain * days

    def _stocking_cost(self) -> float:
        """Costo total de siembra (post-larvas)."""
        return self._total_shrimp_stocked() * self.post_larva_cost

    def _feed_cost(self, days: Optional[int] = None, mortality_rate: float = 0.0) -> float:
        """
        Costo total de alimento durante el ciclo.

        Args:
            days: Días de cultivo.
            mortality_rate: Tasa de mortalidad promedio (fracción 0-1).

        Returns:
            Costo total de alimento ($).
        """
        days = days or self.cycle_days
        total_shrimp = self._total_shrimp_stocked()
        surviving = total_shrimp * (1.0 - mortality_rate)

        total_feed_kg = 0.0
        for day in range(1, days + 1):
            avg_weight_g = self.INITIAL_WEIGHT + self.weight_gain * day
            avg_weight_kg = avg_weight_g / 1000.0
            daily_feed = surviving * avg_weight_kg * self.FEED_RATIO
            total_feed_kg += daily_feed

        return total_feed_kg * self.feed_cost

    def _revenue(self, days: Optional[int] = None, mortality_rate: float = 0.0) -> float:
        """
        Ingreso por venta de camarón.

        Args:
            days: Días de cultivo.
            mortality_rate: Tasa de mortalidad promedio (fracción 0-1).

        Returns:
            Ingreso total ($).
        """
        days = days or self.cycle_days
        total_shrimp = self._total_shrimp_stocked()
        surviving = total_shrimp * (1.0 - mortality_rate)
        avg_weight_kg = self._average_adult_weight(days) / 1000.0

        total_biomass_kg = surviving * avg_weight_kg
        return total_biomass_kg * self.sale_price

    def _action_cost(self, action: int, duration_days: float) -> float:
        """
        Costo de una intervención durante `duration_days` días.

        Args:
            action: Acción ejecutada (0-4).
            duration_days: Duración en días.

        Returns:
            Costo total de la intervención ($).
        """
        daily = self.ACTION_DAILY_COSTS.get(action, 0.0)

        # Si se reduce alimentación, hay ahorro proporcional
        if action == 2:
            # Ahorro aproximado: 20% del costo de alimentación diario
            total_shrimp = self._total_shrimp_stocked()
            avg_weight_kg = self._average_adult_weight() / 1000.0
            daily_feed = total_shrimp * avg_weight_kg * self.FEED_RATIO
            daily_saving = daily_feed * self.feed_cost * 0.20
            return -daily_saving * duration_days  # negativo = ahorro

        return daily * duration_days

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def calculate_scenario(
        self,
        action: int,
        duration_days: float,
        mortality_rate: float,
    ) -> Dict[str, Any]:
        """
        Calcula el escenario económico completo para una intervención.

        Args:
            action: Acción de intervención (0-4).
            duration_days: Duración de la intervención en días.
            mortality_rate: Tasa de mortalidad promedio esperada (0-1).

        Returns:
            Diccionario con revenue, costs, profit, ROI y detalles.
        """
        # Costos
        stocking = self._stocking_cost()
        feed = self._feed_cost(self.cycle_days, mortality_rate)
        intervention = self._action_cost(action, duration_days)

        total_costs = stocking + feed + intervention

        # Ingresos
        revenue = self._revenue(self.cycle_days, mortality_rate)

        # Beneficio y ROI
        profit = revenue - total_costs
        roi = (profit / total_costs * 100.0) if total_costs > 0 else 0.0

        return {
            "action": action,
            "action_name": self.ACTION_NAMES.get(action, "Desconocida"),
            "duration_days": duration_days,
            "mortality_rate": mortality_rate,
            "revenue": round(revenue, 2),
            "stocking_cost": round(stocking, 2),
            "feed_cost": round(feed, 2),
            "intervention_cost": round(intervention, 2),
            "total_costs": round(total_costs, 2),
            "profit": round(profit, 2),
            "roi_percentage": round(roi, 2),
            "total_shrimp_stocked": self._total_shrimp_stocked(),
            "average_weight_g": round(self._average_adult_weight(), 2),
        }

    def compare_policies(self, policy_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compara múltiples escenarios de política en un DataFrame.

        Args:
            policy_results: Lista de diccionarios (salida de calculate_scenario).

        Returns:
            DataFrame con columnas comparativas.
        """
        if not policy_results:
            return pd.DataFrame()

        df = pd.DataFrame(policy_results)

        # Seleccionar y ordenar columnas relevantes
        columns = [
            "action_name", "duration_days", "mortality_rate",
            "revenue", "total_costs", "profit", "roi_percentage",
        ]
        available = [c for c in columns if c in df.columns]
        df = df[available]

        # Renombrar a español
        rename_map = {
            "action_name": "Política",
            "duration_days": "Duración (días)",
            "mortality_rate": "Mortalidad",
            "revenue": "Ingresos ($)",
            "total_costs": "Costos ($)",
            "profit": "Beneficio ($)",
            "roi_percentage": "ROI (%)",
        }
        df = df.rename(columns=rename_map)

        return df

    def simulate_intervention(
        self,
        pre_intervention_mortality: float,
        post_intervention_mortality: float,
        action: int,
        duration: float,
    ) -> Dict[str, Any]:
        """
        Simula el impacto económico de una intervención específica.

        Compara el escenario sin intervención vs. con intervención.

        Args:
            pre_intervention_mortality: Mortalidad esperada sin intervención (0-1).
            post_intervention_mortality: Mortalidad esperada tras la intervención (0-1).
            action: Acción ejecutada (0-4).
            duration: Duración de la intervención en días.

        Returns:
            Diccionario con pérdidas evitadas, costo de acción, beneficio neto y ROI.
        """
        # Escenario sin intervención
        baseline = self.calculate_scenario(0, 0, pre_intervention_mortality)

        # Escenario con intervención
        intervened = self.calculate_scenario(action, duration, post_intervention_mortality)

        losses_avoided = intervened["revenue"] - baseline["revenue"]
        # Restamos también la diferencia en costos de alimentación (menos mortalidad = más comida)
        feed_diff = intervened["feed_cost"] - baseline["feed_cost"]
        action_cost = intervened["intervention_cost"]

        net_benefit = losses_avoided - feed_diff - action_cost

        # ROI de la intervención
        investment = abs(action_cost) + abs(feed_diff) if (action_cost + feed_diff) != 0 else 1.0
        roi_intervention = (net_benefit / investment * 100.0) if investment > 0 else 0.0

        return {
            "action": action,
            "action_name": self.ACTION_NAMES.get(action, "Desconocida"),
            "pre_intervention_mortality": pre_intervention_mortality,
            "post_intervention_mortality": post_intervention_mortality,
            "mortality_reduction": pre_intervention_mortality - post_intervention_mortality,
            "baseline_revenue": round(baseline["revenue"], 2),
            "intervened_revenue": round(intervened["revenue"], 2),
            "losses_avoided": round(losses_avoided, 2),
            "feed_cost_difference": round(feed_diff, 2),
            "action_cost": round(action_cost, 2),
            "net_benefit": round(net_benefit, 2),
            "roi_percentage": round(roi_intervention, 2),
        }

    def generate_report(self, scenarios: List[Dict[str, Any]]) -> str:
        """
        Genera un informe formateado en español a partir de escenarios.

        Args:
            scenarios: Lista de diccionarios (salida de calculate_scenario o
                       simulate_intervention).

        Returns:
            Cadena de texto con el informe formateado.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("  REPORTE ECONÓMICO - CULTIVO DE CAMARÓN")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Área del estanque:      {self.pond_area:,.0f} m²")
        lines.append(f"  Densidad de siembra:    {self.density:.0f} camarones/m²")
        lines.append(f"  Camarones sembrados:    {self._total_shrimp_stocked():,}")
        lines.append(f"  Ciclo de cultivo:       {self.cycle_days} días")
        lines.append(f"  Precio de venta:        ${self.sale_price:.2f}/kg")
        lines.append(f"  Costo de alimento:      ${self.feed_cost:.2f}/kg")
        lines.append(f"  Costo de post-larva:    ${self.post_larva_cost:.4f}/unidad")
        lines.append("")

        for i, sc in enumerate(scenarios, 1):
            lines.append("-" * 70)
            lines.append(f"  Escenario {i}: {sc.get('action_name', 'N/A')}")
            lines.append("-" * 70)

            if "losses_avoided" in sc:
                # Formato de simulate_intervention
                lines.append(f"  Mortalidad pre-intervención:  {sc['pre_intervention_mortality']:.1%}")
                lines.append(f"  Mortalidad post-intervención: {sc['post_intervention_mortality']:.1%}")
                lines.append(f"  Reducción de mortalidad:      {sc['mortality_reduction']:.1%}")
                lines.append(f"  Pérdidas evitadas:            ${sc['losses_avoided']:,.2f}")
                lines.append(f"  Costo de la acción:           ${sc['action_cost']:,.2f}")
                lines.append(f"  Beneficio neto:               ${sc['net_benefit']:,.2f}")
                lines.append(f"  ROI de la intervención:       {sc['roi_percentage']:.2f}%")
            else:
                # Formato de calculate_scenario
                lines.append(f"  Duración:                {sc.get('duration_days', 0):.1f} días")
                lines.append(f"  Mortalidad:              {sc.get('mortality_rate', 0):.1%}")
                lines.append(f"  Ingresos:                ${sc.get('revenue', 0):,.2f}")
                lines.append(f"  Costo de siembra:        ${sc.get('stocking_cost', 0):,.2f}")
                lines.append(f"  Costo de alimento:       ${sc.get('feed_cost', 0):,.2f}")
                lines.append(f"  Costo de intervención:   ${sc.get('intervention_cost', 0):,.2f}")
                lines.append(f"  Costos totales:          ${sc.get('total_costs', 0):,.2f}")
                lines.append(f"  Beneficio:               ${sc.get('profit', 0):,.2f}")
                lines.append(f"  ROI:                     {sc.get('roi_percentage', 0):.2f}%")

            lines.append("")

        lines.append("=" * 70)
        lines.append("  Fin del reporte")
        lines.append("=" * 70)

        return "\n".join(lines)
