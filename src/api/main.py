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

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .mock_data import MockDataManager
from .mqtt_bridge import MqttBridge, bridge_from_env
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
from .stream import SensorStreamer, sse_event_stream

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
mqtt_bridge: Optional[MqttBridge] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa y limpia recursos del ciclo de vida de la aplicación."""
    global manager, mqtt_bridge
    manager = MockDataManager()
    app.state.manager = manager
    mqtt_bridge = bridge_from_env()
    if mqtt_bridge:
        await mqtt_bridge.start()
        app.state.mqtt_bridge = mqtt_bridge
    yield
    if mqtt_bridge:
        await mqtt_bridge.stop()
    manager = None
    mqtt_bridge = None


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


# ---------------------------------------------------------------------------
# Streaming en tiempo real
# ---------------------------------------------------------------------------


@app.websocket("/ws/pond/{pond_id}")
async def ws_pond_stream(websocket: WebSocket, pond_id: str) -> None:
    """
    WebSocket de lectura en vivo de una piscina.

    Envía un mensaje inicial con el estado actual y luego una lectura
    cada `interval` segundos (query param, por defecto 5 s):

        {"type": "reading", "timestamp": ..., "pond_id": ...,
         "species": ..., "sensors": {...}, "outbreak_probability": ...}

    Si hay un puente MQTT activo, las lecturas reales de sensores IoT
    tienen prioridad sobre el streamer simulado.
    """
    mgr = _get_manager()
    if not mgr.pond_exists(pond_id):
        await websocket.close(code=4404)
        return

    await websocket.accept()
    interval = float(websocket.query_params.get("interval", "5"))
    species = websocket.query_params.get("species", "vannamei")

    # Mensaje inicial: estado actual
    status = mgr.get_pond_status(pond_id)
    if status is not None:
        await websocket.send_json({
            "type": "status",
            "pond_id": pond_id,
            "sensors": status.sensors.model_dump(mode="json") if hasattr(status.sensors, "model_dump") else str(status.sensors),
            "alert_level": str(status.alert_level.value) if hasattr(status.alert_level, "value") else str(status.alert_level),
        })

    streamer = SensorStreamer(mgr, pond_id, interval=interval, species=species)
    try:
        if mqtt_bridge is not None and mqtt_bridge.running:
            # Modo MQTT: reenviar lecturas reales
            async for message in mqtt_bridge.readings():
                if message.get("pond_id") == pond_id:
                    await websocket.send_json(message)
        else:
            # Modo simulado: generador interno
            async for message in streamer.readings():
                await websocket.send_json(message)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # pragma: no cover
        logger = __import__("logging").getLogger("bioseguridad.stream")
        logger.warning("Stream de %s terminado: %s", pond_id, exc)


@app.get("/stream/pond/{pond_id}", tags=["Streaming"])
async def sse_pond_stream(
    pond_id: str,
    interval: float = Query(5.0, ge=1.0, le=60.0, description="Segundos entre lecturas"),
    species: str = Query("vannamei", description="Especie para umbrales"),
) -> StreamingResponse:
    """
    Stream de lecturas por Server-Sent Events (SSE).

    Alternativa unidireccional al WebSocket, útil para clientes que no
    necesitan enviar comandos. Consumible con EventSource del navegador.
    """
    mgr = _get_manager()
    if not mgr.pond_exists(pond_id):
        raise HTTPException(status_code=404, detail=f"Estanque '{pond_id}' no encontrado.")

    streamer = SensorStreamer(mgr, pond_id, interval=interval, species=species)
    return StreamingResponse(
        sse_event_stream(streamer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
