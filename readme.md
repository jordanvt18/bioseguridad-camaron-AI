# 🦐 Simulador de Bioseguridad Inteligente para Camarón

Sistema de predicción de brotes de enfermedad en piscinas de camarón mediante **Deep Learning + Reinforcement Learning**, con pipeline ETL, API REST, simulador económico y dashboard operativo.

## Características

- 🔮 **Detección temprana de brotes** con LSTM/Transformer, Autoencoder y clasificador ensemble (RandomForest + One-Class SVM)
- 🎯 **Recomendaciones óptimas** mediante agente RL (PPO, Stable Baselines3) con 5 acciones de mitigación
- 💰 **Simulador económico** con cálculo de pérdidas evitadas, ROI y comparación de políticas
- 📊 **Pipeline ETL** con resampleo a 15 min, detección de fallas de sensores y etiquetado de brotes
- 🌐 **API REST** con FastAPI (7 endpoints) y CORS
- 📈 **Dashboard operativo** (HTML/JS/Plotly) con 4 pestañas: monitoreo, mapa de riesgo, recomendaciones y simulador económico
- 🐳 **Despliegue con Docker** — api (FastAPI), web (nginx) y redis

## Estado del modelo entrenado

Modelo entrenado con **datos reales de producción de camarón (472 registros de 174 piscinas)** + datos de sensores generados con rangos de literatura científica (Islam 2023, Suasono 2025, Kajornkasirat 2021).

| Métrica | Valor | Criterio de aceptación |
|---------|-------|------------------------|
| **AUC** | **0.986** | > 0.85 ✅ |
| **F1** | **0.953** | — |
| **Precision** | **0.997** | — |
| **Precision@10%** | **0.997** | — |
| **Accuracy** | **0.983** | — |

Modelos guardados en `models/`: `rf_shrimp_real.pkl`, `ocsvm_shrimp_real.pkl`, `model_info.json`.

## Inicio Rápido

### Con Docker (recomendado)

```bash
# Clonar el repositorio
git clone <repo-url>
cd bioseguridad-camaron-AI

# Copiar variables de entorno
cp .env.example .env

# Levantar todos los servicios
docker-compose up -d

# Verificar
curl http://localhost:8000/
```

Servicios disponibles:
- **API:** http://localhost:8000
- **Dashboard:** http://localhost:80
- **Redis:** localhost:6379

### Instalación Manual

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env

# Ejecutar API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Dashboard

Abrir `web/index.html` directamente en el navegador (versión estática con datos simulados), o servirlo vía el contenedor nginx de docker-compose.

### ¿Cómo agrego datos reales de mis piscinas?

El dashboard tiene la pestaña **📥 Importar Datos** con 3 vías:

1. **CSV** — descarga la plantilla desde el dashboard (botón *Descargar plantilla CSV*), completa tus lecturas y sube el archivo (arrastrar o clic). Columnas requeridas:

   ```
   timestamp,pond_id,ph,dissolved_oxygen,salinity,turbidity,temperature,ammonia,outbreak
   2025-06-01 06:00,P-001,7.82,5.21,30.1,24.5,29.1,0.35,0
   ```

2. **Entrada manual** — formulario para registrar una lectura puntual de una piscina.

3. **API** — pega la URL de tu API FastAPI desplegada (requiere CORS, ya habilitado) para sincronizar el estado.

Los datos se validan fila por fila (rangos físicos), se guardan en el navegador (localStorage) y alimentan todos los gráficos, el mapa de riesgo y las recomendaciones. La cabecera muestra la fuente: *Simulada* o *Real (N registros)*.

### Especies soportadas

El selector de especie en la cabecera ajusta umbrales de riesgo, curva de crecimiento, tallas y precio. Incluye las especies de Ecuador y extrapolación a otras especies marinas:

| Especie | Científico | Temp. óptima | Origen |
|---------|-----------|--------------|--------|
| 🦐 Camarón Blanco | *Litopenaeus vannamei* | 26–32 °C | 🇪🇨 Ecuador (dominante) |
| 🦐 Camarón Azul | *Penaeus stylirostris* | 24–30 °C | Pacífico americano |
| 🦐 Camarón Tigre | *Penaeus monodon* | 27–32 °C | Indo-Pacífico |
| 🐟 Tilapia | *Oreochromis niloticus* | 25–32 °C | Extrapolación |
| 🐟 Trucha | *Oncorhynchus mykiss* | 10–18 °C | Extrapolación |
| 🐟 Salmón | *Salmo salar* | 6–14 °C | Extrapolación |
| 🐟 Corvina/Róbalo | *Centropomus viridis* | 24–30 °C | Pacífico ecuatoriano |

La pestaña **🦐 Especies** muestra el catálogo completo, parámetros óptimos, tallas comerciales ecuatorianas (colas por libra: U/10 → 91/120) y la curva de crecimiento proyectada.

## Pipeline de datos (ETL)

```bash
# Generar datos sintéticos y procesar (sensores, brotes, operaciones, clima)
python scripts/run_etl.py --n-ponds 5 --days 90

# Integrar datos reales de producción + generar sensores con rangos de literatura
python scripts/integrate_real_data.py
```

Salida: `data/processed/merged_features_real.parquet` (43,200 registros, 20.4% brotes).

## Entrenar modelos

```bash
# Entrenar detector de brotes (RandomForest + One-Class SVM)
python scripts/train_real_data.py

# Entrenar agente RL (PPO)
python src/rl/train_agent.py --algorithm ppo --timesteps 100000
```

## Estructura del Proyecto

```
bioseguridad-camaron-AI/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── METHODOLOGY.md
├── README.md
├── data/
│   ├── raw/              # CSV crudos + datos reales de producción
│   └── processed/        # Parquet procesados
├── models/               # Modelos entrenados
├── scripts/
│   ├── run_etl.py
│   ├── integrate_real_data.py
│   └── train_real_data.py
├── src/
│   ├── etl/              # etl_sensors, etl_brotes, etl_operations, etl_climate
│   ├── models/           # lstm, transformer, autoencoder, classifier, train
│   ├── rl/               # environment, train_agent, economic_simulator, policy_recommender
│   └── api/              # main, schemas, mock_data
├── tests/                # 124 tests (ETL 33 + DL 22 + RL 33 + API 36)
└── web/                  # Dashboard operativo (index.html, nginx.conf)
```

## API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Health check del sistema |
| GET | `/ponds` | Lista de piscinas con estado |
| GET | `/pond/{id}/status` | Sensores y estado actual de la piscina |
| GET | `/pond/{id}/risk` | Probabilidad de brote (0–1) y nivel de riesgo |
| POST | `/pond/{id}/simulate` | Simulación económica de una intervención |
| GET | `/policy/recommendation?pond_id={id}` | Acción recomendada por el agente RL |
| GET | `/farm/{id}/risk-map` | Mapa de riesgo de todas las piscinas de una granja |

Documentación interactiva: `http://localhost:8000/docs` (Swagger UI).

## Configuración

Copiar `.env.example` a `.env` y ajustar valores:

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `API_PORT` | Puerto de la API | `8000` |
| `REDIS_URL` | URL de conexión a Redis | `redis://redis:6379` |
| `MODEL_PATH` | Ruta a modelos entrenados | `./models` |
| `DATA_PATH` | Ruta a datos | `./data` |
| `LOG_LEVEL` | Nivel de logging | `INFO` |
| `MQTT_ENABLED` | Activar puente MQTT (`1`/`0`) | `0` |
| `MQTT_BROKER` | Host del broker MQTT | `localhost` |
| `MQTT_PORT` | Puerto del broker MQTT | `1883` |
| `MQTT_USERNAME` / `MQTT_PASSWORD` | Credenciales del broker | — |

## Streaming de datos en tiempo real

El sistema soporta datos reales por streaming con esta arquitectura:

```
[Sensores IoT (ESP32/Arduino)] --MQTT--> [Broker Mosquitto/EMQX]
                                              |
                                        paho-mqtt
                                              v
                              [MqttBridge: bioseguridad/sensors/{pond_id}]
                                              |
                                        WebSocket
                                              v
                              [API FastAPI] <--> [Dashboard operativo]
```

### Endpoints de streaming

| Transporte | Ruta | Descripción |
|------------|------|-------------|
| WebSocket | `/ws/pond/{pond_id}?interval=5&species=vannamei` | Lecturas en vivo (recomendado, bidireccional) |
| SSE | `/stream/pond/{pond_id}?interval=5&species=vannamei` | Alternativa unidireccional (EventSource) |

Cada mensaje (`type: "reading"`) incluye: `timestamp`, `pond_id`, `species`, `sensors` (ph, dissolved_oxygen, salinity, turbidity, temperature, ammonia) y `outbreak_probability` (0–1).

### Sensores reales por MQTT

1. Instala un broker (Mosquitto) y configura `.env` con `MQTT_ENABLED=1`.
2. Los sensores publican JSON en `bioseguridad/sensors/{pond_id}`:

   ```json
   {"ph": 7.8, "dissolved_oxygen": 5.2, "salinity": 30.1, "turbidity": 25.0, "temperature": 29.1, "ammonia": 0.35}
   ```

3. El dashboard se conecta con el botón **🔴 Streaming en vivo** (pestaña Monitoreo): `ws://localhost:8000/ws/pond/pond-001`. Incluye reconexión automática con backoff y actualiza KPIs, sensores, riesgo y gráficos en tiempo real.

> **Nota:** GitHub Pages sirve solo archivos estáticos (no WebSockets). El streaming requiere la API corriendo — localmente con `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`, o desplegada en un servicio con soporte WebSocket (Render/Fly/Railway).

## Testing

```bash
# Ejecutar todos los tests
pytest

# Tests específicos
pytest tests/test_etl.py -v
pytest tests/test_models.py -v
pytest tests/test_rl.py -v
pytest tests/test_api.py -v
```

## Despliegue

### Dashboard (Function Compute / nginx estático)

El dashboard es estático y está preparado para Function Compute con `nginx.conf` (listen 9000, root /code). El zip del sitio se genera con el contenido de `web/` + `nginx.conf` en la raíz.

### API (Docker)

```bash
docker-compose up -d api
```

## Fuentes de datos

- **Datos reales de producción**: [SHRIMP_POND_MONITORING](https://github.com/Aryanyadav123456/SHRIMP_POND_MONITORING) — 472 registros, 174 piscinas
- **Rangos de sensores (literatura)**: Islam et al. (2023), Suasono et al. (2025), Kajornkasirat et al. (2021)
- Los datos crudos están en `data/raw/` con atribución en `models/model_info.json`

## Licencia

MIT License — ver archivo [LICENSE](LICENSE) para detalles.

---

*Desarrollado para la acuicultura sostenible 🦐*
