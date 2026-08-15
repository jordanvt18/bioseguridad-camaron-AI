"""
Aplicación FastAPI: Simulador de Bioseguridad Inteligente para Camarón.

API REST que sirve predicciones de riesgo de brote, estado de estanques,
simulaciones económicas y recomendaciones de política de manejo.

Diseñada para funcionar de forma independiente usando datos simulados
cuando los modelos DL/RL no están disponibles.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .mock_data import MockDataManager
from .schemas import (
    FarmRiskMap,
    HealthResponse,
    PondRiskSummary,
    PondStatus,
    PolicyRecommendation,
    RiskAssessment,
    RiskLevel,
    SimulationRequest,
    SimulationResult,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

API_VERSION = "1.0.0"
API_TITLE = "Simulador de Bioseguridad Inteligente para Camarón"
API_DESCRIPTION = (
    "Sistema de predicción de brotes de enfermedad en estanques camaroneros "
    "usando Deep Learning y Reinforcement Learning. Esta API proporciona "
    "evaluaciones de riesgo, estado de estanques, simulaciones económicas "
    "y recomendaciones de política de manejo."
)

# ---------------------------------------------------------------------------
# Gestor de datos (singleton a nivel de módulo)
# ---------------------------------------------------------------------------

manager: Optional[MockDataManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa y limpia recursos del ciclo de vida de la aplicación."""
    global manager
    manager = MockDataManager()
    app.state.manager = manager
    yield
    manager = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_manager() -> MockDataManager:
    """Obtiene el gestor de datos activo."""
    if manager is None:
        raise RuntimeError("El gestor de datos no está inicializado.")
    return manager


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Sistema"])
async def health_check() -> HealthResponse:
    """
    Health check del sistema.

    Retorna información básica del servicio incluyendo versión,
    estado del modelo y número de estanques monitoreados.
    """
    mgr = _get_manager()
    return HealthResponse(
        status="operativo",
        version=API_VERSION,
        model_loaded=False,
        n_ponds=mgr.n_ponds,
    )


@app.get("/ponds", response_model=list[PondStatus], tags=["Estanques"])
async def list_ponds() -> list[PondStatus]:
    """
    Lista todos los estanques monitoreados con su estado actual.

    Retorna el estado de los 15 estanques distribuidos en 3 granjas,
    incluyendo lecturas de sensores y nivel de alerta.
    """
    mgr = _get_manager()
    return mgr.get_all_ponds()


@app.get(
    "/pond/{pond_id}/status",
    response_model=PondStatus,
    tags=["Estanques"],
)
async def get_pond_status(pond_id: str) -> PondStatus:
    """
    Obtiene el estado actual de un estanque específico.

    Incluye lecturas de sensores de calidad del agua, nivel de alerta
    y última actualización.

    Args:
        pond_id: Identificador único del estanque (ej. pond-001).
    """
    mgr = _get_manager()
    status = mgr.get_pond_status(pond_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Estanque '{pond_id}' no encontrado.",
        )
    return status


@app.get(
    "/pond/{pond_id}/risk",
    response_model=RiskAssessment,
    tags=["Riesgo"],
)
async def get_pond_risk(pond_id: str) -> RiskAssessment:
    """
    Evalúa el riesgo de brote de enfermedad para un estanque.

    Retorna la probabilidad de brote (0-1), nivel de riesgo categorizado
    (bajo/medio/alto/crítico) y los factores contribuyentes identificados.

    Args:
        pond_id: Identificador único del estanque.
    """
    mgr = _get_manager()
    risk = mgr.simulate_outbreak_probability(pond_id)
    if risk is None:
        raise HTTPException(
            status_code=404,
            detail=f"Estanque '{pond_id}' no encontrado.",
        )
    return risk


@app.post(
    "/pond/{pond_id}/simulate",
    response_model=SimulationResult,
    tags=["Simulación"],
)
async def simulate_action(
    pond_id: str,
    request: SimulationRequest,
) -> SimulationResult:
    """
    Simula el resultado económico de una acción de manejo.

    Permite evaluar el impacto económico de diferentes acciones
    (mantener, recambio de agua, reducir densidad, probióticos,
    cosecha parcial) sobre un período de tiempo determinado.

    Args:
        pond_id: Identificador único del estanque.
        request: Acción a simular (0-4) y duración en días.
    """
    mgr = _get_manager()
    if not mgr.pond_exists(pond_id):
        raise HTTPException(
            status_code=404,
            detail=f"Estanque '{pond_id}' no encontrado.",
        )
    result = mgr.simulate_action(pond_id, request.action, request.duration_days)
    if result is None:
        raise HTTPException(
            status_code=500,
            detail="Error al generar la simulación.",
        )
    return SimulationResult(**result)


@app.get(
    "/policy/recommendation",
    response_model=PolicyRecommendation,
    tags=["Política"],
)
async def get_policy_recommendation(
    pond_id: str = Query(..., description="Identificador del estanque"),
) -> PolicyRecommendation:
    """
    Obtiene la recomendación de política de manejo para un estanque.

    Basado en el riesgo actual, recomienda una acción de manejo
    con su nivel de confianza, resultado esperado e impacto económico.

    Args:
        pond_id: Identificador único del estanque (query param).
    """
    mgr = _get_manager()
    if not mgr.pond_exists(pond_id):
        raise HTTPException(
            status_code=404,
            detail=f"Estanque '{pond_id}' no encontrado.",
        )
    rec = mgr.get_policy_recommendation(pond_id)
    if rec is None:
        raise HTTPException(
            status_code=500,
            detail="Error al generar la recomendación.",
        )
    return PolicyRecommendation(**rec)


@app.get(
    "/farm/{farm_id}/risk-map",
    response_model=FarmRiskMap,
    tags=["Granjas"],
)
async def get_farm_risk_map(farm_id: str) -> FarmRiskMap:
    """
    Obtiene el mapa de riesgo de todos los estanques de una granja.

    Retorna un resumen de riesgo para cada estanque de la granja,
    incluyendo probabilidad de brote y nivel de riesgo categorizado,
    junto con un riesgo general de la granja.

    Args:
        farm_id: Identificador único de la granja (ej. farm-001).
    """
    mgr = _get_manager()
    if not mgr.farm_exists(farm_id):
        raise HTTPException(
            status_code=404,
            detail=f"Granja '{farm_id}' no encontrada.",
        )

    risks = mgr.get_farm_risks(farm_id)
    if risks is None:
        raise HTTPException(
            status_code=500,
            detail="Error al generar el mapa de riesgo.",
        )

    pond_summaries = [
        PondRiskSummary(
            pond_id=r.pond_id,
            name=mgr.get_pond_status(r.pond_id).name if mgr.get_pond_status(r.pond_id) else r.pond_id,
            outbreak_probability=r.outbreak_probability,
            risk_level=r.risk_level,
        )
        for r in risks
    ]

    # Riesgo general: el máximo nivel de los estanques
    max_prob = max(r.outbreak_probability for r in risks)
    if max_prob >= 0.7:
        overall = RiskLevel.CRITICO
    elif max_prob >= 0.5:
        overall = RiskLevel.ALTO
    elif max_prob >= 0.3:
        overall = RiskLevel.MEDIO
    else:
        overall = RiskLevel.BAJO

    return FarmRiskMap(
        farm_id=farm_id,
        farm_name=mgr.get_farm_name(farm_id) or farm_id,
        ponds=pond_summaries,
        overall_risk=overall,
    )
