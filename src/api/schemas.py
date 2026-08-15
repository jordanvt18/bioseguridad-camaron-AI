"""
Esquemas Pydantic v2 para la API de Bioseguridad Inteligente para Camarón.

Define todos los modelos de datos para lecturas de sensores, estado de estanques,
evaluación de riesgo, simulación económica y recomendaciones de política.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumeraciones
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Nivel de riesgo de brote de enfermedad."""
    BAJO = "bajo"
    MEDIO = "medio"
    ALTO = "alto"
    CRITICO = "crítico"


class AlertLevel(str, Enum):
    """Nivel de alerta operacional del estanque."""
    NORMAL = "normal"
    PRECAUCION = "precaución"
    ALERTA = "alerta"
    EMERGENCIA = "emergencia"


# ---------------------------------------------------------------------------
# Lecturas de sensores
# ---------------------------------------------------------------------------

class SensorReading(BaseModel):
    """Lectura individual de sensores de calidad del agua."""
    ph: float = Field(
        ..., ge=0, le=14,
        description="Potencial de hidrógeno (0-14)"
    )
    dissolved_oxygen: float = Field(
        ..., ge=0, le=20,
        description="Oxígeno disuelto en mg/L"
    )
    salinity: float = Field(
        ..., ge=0, le=50,
        description="Salinidad en ppt (partes por mil)"
    )
    turbidity: float = Field(
        ..., ge=0, le=100,
        description="Turbidez en NTU"
    )
    temperature: float = Field(
        ..., ge=0, le=45,
        description="Temperatura del agua en °C"
    )
    ammonia: float = Field(
        ..., ge=0, le=10,
        description="Concentración de amoníaco en mg/L"
    )
    timestamp: datetime = Field(
        ..., description="Momento de la lectura (ISO 8601)"
    )


# ---------------------------------------------------------------------------
# Estado del estanque
# ---------------------------------------------------------------------------

class PondStatus(BaseModel):
    """Estado actual de un estanque con lecturas de sensores."""
    pond_id: str = Field(..., description="Identificador único del estanque")
    farm_id: str = Field(..., description="Identificador de la granja")
    name: str = Field(..., description="Nombre del estanque")
    sensors: SensorReading = Field(..., description="Lecturas actuales de sensores")
    alert_level: AlertLevel = Field(..., description="Nivel de alerta operacional")
    last_update: datetime = Field(..., description="Última actualización")


# ---------------------------------------------------------------------------
# Evaluación de riesgo
# ---------------------------------------------------------------------------

class ContributingFactor(BaseModel):
    """Factor contribuyente al riesgo de brote."""
    factor: str = Field(..., description="Nombre del factor")
    value: float = Field(..., description="Valor actual del factor")
    optimal_range: str = Field(..., description="Rango óptimo del factor")
    severity: float = Field(
        ..., ge=0, le=1,
        description="Severidad de desviación (0-1)"
    )


class RiskAssessment(BaseModel):
    """Evaluación de riesgo de brote de enfermedad."""
    pond_id: str = Field(..., description="Identificador del estanque")
    outbreak_probability: float = Field(
        ..., ge=0, le=1,
        description="Probabilidad de brote (0-1)"
    )
    risk_level: RiskLevel = Field(..., description="Nivel de riesgo categorizado")
    contributing_factors: list[ContributingFactor] = Field(
        ..., description="Factores que contribuyen al riesgo"
    )
    timestamp: datetime = Field(..., description="Momento de la evaluación")


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    """Petición de simulación de acción de manejo."""
    action: int = Field(
        ..., ge=0, le=4,
        description=(
            "Acción a simular: 0=mantener, 1=recambio de agua, "
            "2=reducir densidad, 3=aplicar probióticos, 4=cosecha parcial"
        )
    )
    duration_days: int = Field(
        ..., ge=1, le=365,
        description="Duración de la simulación en días"
    )


class SimulationResult(BaseModel):
    """Resultado de simulación económica de una acción de manejo."""
    action_taken: int = Field(..., description="Acción simulada (0-4)")
    action_name: str = Field(..., description="Nombre descriptivo de la acción")
    duration_days: int = Field(..., description="Duración simulada en días")
    revenue: float = Field(..., description="Ingresos proyectados (USD)")
    costs: float = Field(..., description="Costos proyectados (USD)")
    profit: float = Field(..., description="Ganancia neta (USD)")
    roi_percentage: float = Field(
        ..., description="Retorno de inversión (%)"
    )
    losses_avoided: float = Field(
        ..., description="Pérdidas evitadas por la acción (USD)"
    )
    net_benefit: float = Field(
        ..., description="Beneficio neto de la acción vs. no actuar (USD)"
    )


# ---------------------------------------------------------------------------
# Recomendación de política
# ---------------------------------------------------------------------------

class PolicyRecommendation(BaseModel):
    """Recomendación de política de manejo generada por RL."""
    pond_id: str = Field(..., description="Identificador del estanque")
    recommended_action: int = Field(
        ..., ge=0, le=4,
        description="Acción recomendada (0-4)"
    )
    action_name: str = Field(..., description="Nombre de la acción recomendada")
    confidence: float = Field(
        ..., ge=0, le=1,
        description="Confianza del modelo en la recomendación (0-1)"
    )
    expected_outcome: str = Field(
        ..., description="Resultado esperado descrito en español"
    )
    economic_impact: float = Field(
        ..., description="Impacto económico proyectado (USD)"
    )


# ---------------------------------------------------------------------------
# Mapa de riesgo de granja
# ---------------------------------------------------------------------------

class PondRiskSummary(BaseModel):
    """Resumen de riesgo de un estanque para el mapa de granja."""
    pond_id: str = Field(..., description="Identificador del estanque")
    name: str = Field(..., description="Nombre del estanque")
    outbreak_probability: float = Field(
        ..., ge=0, le=1,
        description="Probabilidad de brote (0-1)"
    )
    risk_level: RiskLevel = Field(..., description="Nivel de riesgo")


class FarmRiskMap(BaseModel):
    """Mapa de riesgo de todos los estanques de una granja."""
    farm_id: str = Field(..., description="Identificador de la granja")
    farm_name: str = Field(..., description="Nombre de la granja")
    ponds: list[PondRiskSummary] = Field(
        ..., description="Lista de estanques con su resumen de riesgo"
    )
    overall_risk: RiskLevel = Field(
        ..., description="Riesgo general de la granja"
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud del sistema."""
    status: str = Field(..., description="Estado del servicio")
    version: str = Field(..., description="Versión de la API")
    model_loaded: bool = Field(
        ..., description="Si el modelo DL/RL está cargado"
    )
    n_ponds: int = Field(
        ..., ge=0,
        description="Número de estanques monitoreados"
    )
