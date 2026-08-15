# Metodología — Simulador de Bioseguridad Inteligente para Camarón

## 1. Resumen Ejecutivo

El **Simulador de Bioseguridad Inteligente para Camarón** es un sistema de software que combina aprendizaje profundo (Deep Learning, DL) y aprendizaje por refuerzo (Reinforcement Learning, RL) para predecir brotes de enfermedad en estanques camaroneros y recomendar acciones de mitigación óptimas. El sistema integra un pipeline ETL para la ingesta y procesamiento de datos de sensores, un modelo de detección temprana basado en LSTM/Transformer, un entorno RL para la toma de decisiones, un simulador económico para la evaluación de políticas, y una API REST con dashboard operativo para usuarios finales.

El objetivo es reducir la mortalidad por enfermedades (como WFD, AHPND, White Spot) mediante alertas tempranas y recomendaciones accionables, minimizando pérdidas económicas en la acuicultura de camarón.

---

## 2. Arquitectura del Sistema

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────┐
│                   Fuentes de Datos                       │
│   (Sensores IoT, API climática, Registros manuales)      │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│   Pipeline ETL        │   │   Almacenamiento (Parquet)    │
│  - Resample 15 min    │──▶│   data/raw/  data/processed/  │
│  - Detección de fallas│   └──────────────┬───────────────┘
│  - Etiquetado brotes  │                  │
└──────────────────────┘                  ▼
                               ┌──────────────────────────┐
                               │  Modelo de Detección (DL) │
                               │  LSTM / Transformer /     │
                               │  Autoencoder              │
                               └──────────────┬───────────┘
                                              │
                                              ▼
                               ┌──────────────────────────┐
                               │   Entorno RL (PPO)        │
                               │   Estado + Recompensa     │
                               └──────────────┬───────────┘
                                              │
                                              ▼
                               ┌──────────────────────────┐
                               │   Simulador Económico     │
                               │   ROI / Comparación       │
                               └──────────────┬───────────┘
                                              │
                                              ▼
                               ┌──────────────────────────┐
                               │   API REST (FastAPI)      │
                               │   Endpoints + WebSocket   │
                               └──────────────┬───────────┘
                                              │
                                              ▼
                               ┌──────────────────────────┐
                               │   Dashboard Operativo     │
                               │   (HTML/JS + Nginx)       │
                               └──────────────────────────┘
```

### Flujo de Datos

1. **Ingesta:** Los datos de sensores (oxígeno disuelto, temperatura, pH, salinidad, turbidez) se reciben en intervalos irregulares y se almacenan en `data/raw/`.
2. **Procesamiento ETL:** El pipeline remuestrea a intervalos de 15 minutos, detecta fallas de sensores, imputa valores faltantes, y etiqueta períodos de brote usando registros histopatológicos.
3. **Almacenamiento:** Los datos procesados se guardan en formato Parquet en `data/processed/` para acceso eficiente.
4. **Inferencia DL:** El modelo de detección temprana consume ventanas de tiempo (lookback=96, equivalente a 24 horas) y produce una probabilidad de brote.
5. **Decisión RL:** El agente PPO recibe el estado del estanque (datos de sensores + probabilidad DL) y selecciona una acción de mitigación.
6. **Simulación Económica:** El simulador calcula el ROI esperado de la acción recomendada versus no actuar.
7. **Presentación:** La API REST expone los resultados; el dashboard los visualiza en tiempo real con alertas.

---

## 3. Pipeline ETL

### Fuentes de Datos

| Fuente | Variables | Frecuencia |
|--------|-----------|------------|
| Sensores IoT (DO, T, pH, salinidad, turbidez) | Oxígeno disuelto, temperatura, pH, salinidad, turbidez | Variable (1–60 min) |
| API Climática | Temperatura ambiente, precipitación, velocidad del viento | Horaria |
| Registros manuales | Observaciones de comportamiento, alimentación, mortalidad | Diaria |

### Procesamiento

1. **Resample a 15 minutos:** Todos los datos se remuestrean a una frecuencia uniforme de 15 minutos usando interpolación lineal para valores numéricos y forward-fill para variables categóricas.
2. **Detección de fallas de sensores:**
   - Identificación de valores fuera de rango físico (ej. pH < 0 o pH > 14).
   - Detección de sensores atascados (varianza ≈ 0 en ventana de 6 horas).
   - Marcado de períodos con fallas para exclusión del entrenamiento.
3. **Imputación:** Valores faltantes se imputan usando interpolación lineal para huecos cortos (< 4 muestras) y mediana móvil para huecos largos.
4. **Etiquetado de brotes:** Se utiliza un período de 48 horas antes de la confirmación histopatológica como ventana de etiqueta positiva, permitiendo que el modelo aprenda patrones tempranos.
5. **Feature engineering:** Se calculan estadísticas rodantes (media, desviación estándar, pendiente) en ventanas de 1h, 6h y 24h para cada variable.

### Almacenamiento

- **Formato:** Apache Parquet (compresión Snappy)
- **Ubicación:** `data/processed/`
- **Particionado:** Por estanque y por mes (`estanque=01/año=2024/mes=06/datos.parquet`)
- **Esquema:** Cada archivo contiene columnas `timestamp`, `estanque_id`, variables de sensores, features derivadas, y `label_brote` (0/1).

---

## 4. Modelo de Detección Temprana (DL)

### 4.1 LSTM — Modelo Principal

**Arquitectura:**

| Capa | Configuración |
|------|---------------|
| Input | (batch, 96, n_features) — lookback de 96 pasos (24h a 15 min) |
| LSTM 1 | 128 unidades, return_sequences=True, dropout=0.2 |
| LSTM 2 | 64 unidades, return_sequences=False |
| Dense | 32, activación ReLU |
| Dropout | 0.3 |
| Output | 1, activación Sigmoide (probabilidad de brote) |

**Entrenamiento:**
- Optimizador: Adam (lr=1e-3, weight_decay=1e-5)
- Función de pérdida: Binary Cross-Entropy con peso de clase (penaliza falsos negativos 3x)
- Early stopping: paciencia de 10 épocas sobre validation AUC
- Batch size: 64
- División: 70% train, 15% validation, 15% test (división temporal, no aleatoria)

### 4.2 Transformer — Arquitectura Alternativa

**Arquitectura:**

| Capa | Configuración |
|------|---------------|
| Input | (batch, 96, n_features) |
| Positional Encoding | Seno/Coseno |
| Transformer Encoder | 4 capas, 8 cabezas, d_model=128, d_ff=512 |
| Global Average Pooling | Sobre la dimensión temporal |
| Dense | 64, activación ReLU |
| Dropout | 0.3 |
| Output | 1, activación Sigmoide |

**Ventajas:** Captura dependencias de largo plazo más eficientemente que LSTM mediante mecanismos de atención. Útil cuando se amplía el lookback más allá de 96 pasos.

### 4.3 Autoencoder — Detección de Anomalías

**Arquitectura:**

- **Encoder:** LSTM(64) → LSTM(32) → vector latente (dim 16)
- **Decoder:** LSTM(32) → LSTM(64) → reconstrucción (dim = n_features)
- **Entrenamiento:** Solo con datos normales (sin brotes), minimizando error de reconstrucción (MSE)
- **Detección:** Un alto error de reconstrucción (umbral percentil 95) indica anomalía

**Uso:** Se combina con el clasificador principal — una anomalía detectada por el autoencoder incrementa la probabilidad de brote del modelo LSTM.

### 4.4 Clasificador de Probabilidad de Brote

El sistema fusiona las predicciones de los tres modelos:

```
P(brote) = α · P(LSTM) + β · P(Transformer) + γ · Anomaly_score(Autoencoder)
```

Donde α, β, γ son pesos aprendidos mediante regresión logística sobre el conjunto de validación (por defecto α=0.5, β=0.3, γ=0.2).

### 4.5 Métricas de Evaluación

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| AUC-ROC | Área bajo curva ROC | ≥ 0.85 |
| F1-Score | Media armónica de precision y recall | ≥ 0.75 |
| Precision@k | Precisión en el top-k% de predicciones de mayor riesgo | ≥ 0.80 (k=10%) |
| Lead Time | Tiempo de anticipación de la alerta antes del brote confirmado | ≥ 6 horas |
| Falsa Alarma | Tasa de falsos positivos por estanque por semana | ≤ 1 |

---

## 5. Entorno RL

### Estado y Espacio de Acciones

**Estado (observación):**

Vector de dimensión variable que incluye:
- Datos de sensores de las últimas 24h (resampleados, n_features × 96)
- Probabilidad de brote del modelo DL (escalar)
- Día del ciclo de cultivo (escalar normalizado)
- Biomasa estimada del estanque (escalar normalizado)
- Historial de acciones tomadas en las últimas 48h

**Espacio de Acciones (5 acciones discretas):**

| Acción | Descripción | Impacto |
|--------|-------------|---------|
| 0 | No intervenir | Sin costo; riesgo de brote no mitigado |
| 1 | Ajuste de alimentación | Reduce estrés; costo bajo |
| 2 | Recambio de agua parcial (20%) | Mejora calidad de agua; costo medio |
| 3 | Aplicación de probióticos | Refuerza sistema inmunológico; costo medio-alto |
| 4 | Cosecha preventiva | Elimina riesgo pero reduce rendimiento; costo alto |

### Función de Recompensa

```
R(s, a) = w1 · Δ_riesgo_brote - w2 · costo(a) + w3 · supervivencia_esperada - w4 · penalización_brote
```

Donde:
- `Δ_riesgo_brote`: Reducción de probabilidad de brote tras la acción
- `costo(a)`: Costo económico de la acción (escala 0–1)
- `supervivencia_esperada`: Tasa de supervivencia proyectada a cosecha
- `penalización_brote`: Penalización grande (-10) si ocurre un brote a pesar de la acción
- Pesos: w1=1.0, w2=0.3, w3=0.5, w4=2.0

### Entrenamiento con PPO

- **Algoritmo:** Proximal Policy Optimization (PPO) de Stable-Baselines3
- **Política:** MlpPolicy con 2 capas ocultas de 64 neuronas
- **Horizonte:** Cada episodio simula un ciclo de cultivo completo (~120 días)
- **Paralelización:** 8 entornos simultáneos (SubprocVecEnv)
- **Hiperparámetros:**
  - Learning rate: 3e-4
  - n_steps: 2048
  - batch_size: 64
  - n_epochs: 10
  - gamma: 0.99
  - clip_range: 0.2
- **Total de timesteps:** 1,000,000 (con evaluación cada 10,000 pasos)
- **Criterio de convergencia:** Recompensa media estabilizada en últimos 50 episodios de evaluación

---

## 6. Simulador Económico

### Modelo de Costos

| Componente | Descripción | Unidad |
|------------|-------------|--------|
| Alimento | Costo de balanced feed por kg | USD/kg |
| Probióticos | Costo por aplicación | USD/ha |
| Energía | Bombeo y aireación | USD/kWh |
| Mano de obra | Operarios y técnicos | USD/día |
| Postlarva | Costo de siembra | USD/1000 postlarvas |
| Combustible | Transporte y mantenimiento | USD/mes |

### Cálculo de ROI

```
ROI = (Ingreso_esperado - Costo_total) / Costo_total × 100
```

Donde:
- **Ingreso_esperado** = Biomasa_cosecha × Precio_camarón/kg
- **Biomasa_cosecha** = supervivencia × peso_promedio × densidad_siembra × área
- **Costo_total** = Σ costos operativos + costo de acciones de mitigación

### Comparación de Políticas

El simulador permite comparar:

1. **Política RL:** Recomendaciones del agente PPO
2. **Política reactiva:** Intervención solo tras detectar mortalidad
3. **Política preventiva estándar:** Protocolo fijo (probióticos semanales, recambios programados)
4. **No intervención:** Línea base sin mitigación

Métricas de comparación: ROI, supervivencia, costo por kg producido, número de brotes evitados.

---

## 7. API REST

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/api/v1/estanques` | Lista de estanques monitoreados |
| GET | `/api/v1/estanques/{id}/sensores` | Datos de sensores actuales |
| GET | `/api/v1/estanques/{id}/riesgo` | Probabilidad de brote actual |
| POST | `/api/v1/estanques/{id}/accion` | Registrar acción tomada |
| GET | `/api/v1/estanques/{id}/recomendacion` | Recomendación del agente RL |
| GET | `/api/v1/estanques/{id}/simulacion` | Resultado de simulación económica |
| GET | `/api/v1/modelo/metricas` | Métricas del modelo DL |
| POST | `/api/v1/etl/run` | Ejecutar pipeline ETL manualmente |
| WebSocket | `/ws/estanques/{id}` | Stream de datos en tiempo real |

### Esquemas de Datos

**SensorData:**
```json
{
  "estanque_id": "string",
  "timestamp": "ISO-8601",
  "oxigeno_disuelto": "float (mg/L)",
  "temperatura": "float (°C)",
  "ph": "float",
  "salinidad": "float (ppm)",
  "turbidez": "float (NTU)"
}
```

**RiesgoBrote:**
```json
{
  "estanque_id": "string",
  "probabilidad": "float [0-1]",
  "nivel_riesgo": "bajo | medio | alto | crítico",
  "lead_time_horas": "float",
  "modelo_version": "string"
}
```

**Recomendacion:**
```json
{
  "estanque_id": "string",
  "accion_recomendada": "int (0-4)",
  "descripcion_accion": "string",
  "roi_esperado": "float (%)",
  "confianza": "float [0-1]"
}
```

---

## 8. Dashboard Operativo

### Visualizaciones

1. **Mapa de estanques:** Vista general con código de color (verde/amarillo/rojo) según nivel de riesgo.
2. **Gráfico temporal de sensores:** Series temporales de las últimas 48h con bandas de normalidad.
3. **Indicador de probabilidad de brote:** Gauge con valor actual y tendencia.
4. **Panel de recomendaciones:** Tarjeta con la acción recomendada y ROI esperado.
5. **Histórico de brotes:** Línea de tiempo de eventos pasados por estanque.
6. **Comparación de políticas:** Gráfico de barras con ROI por política.

### Alertas en Tiempo Real

| Nivel | Condición | Acción del Sistema |
|-------|-----------|--------------------|
| Bajo | P(brote) < 0.2 | Monitoreo normal |
| Medio | 0.2 ≤ P(brote) < 0.5 | Notificación en dashboard |
| Alto | 0.5 ≤ P(brote) < 0.8 | Notificación push + recomendación de acción |
| Crítico | P(brote) ≥ 0.8 | Alerta sonora + notificación prioritaria + activación automática de simulación |

Las alertas se entregan vía WebSocket a los clientes conectados y se registran en el log del sistema.

---

## 9. Criterios de Aceptación

| # | Criterio | Verificación |
|---|----------|--------------|
| AC-1 | El pipeline ETL procesa 30 días de datos de 10 estanques en < 5 minutos | Test de rendimiento |
| AC-2 | El modelo LSTM alcanza AUC ≥ 0.85 en conjunto de test | Evaluación cuantitativa |
| AC-3 | El sistema genera alertas con ≥ 6 horas de anticipación al brote | Métrica de lead time |
| AC-4 | El agente RL supera la política reactiva en ROI en simulación | Comparación estadística |
| AC-5 | La API responde en < 200ms (p95) para endpoints de consulta | Test de carga |
| AC-6 | El dashboard muestra datos actualizados en < 5 segundos | Test de integración |
| AC-7 | La tasa de falsas alarmas es ≤ 1 por estanque por semana | Métrica operacional |
| AC-8 | El sistema soporta 50 estanques concurrentes | Test de escalabilidad |

---

## 10. Limitaciones y Trabajo Futuro

### Limitaciones Actuales

1. **Dependencia de datos etiquetados:** El modelo supervisado requiere registros histopatológicos confirmados, que son escasos.
2. **Generalización entre regiones:** El modelo entrenado en una zona geográfica puede no generalizar a otras con patógenos o condiciones diferentes.
3. **Sensores como única fuente:** No se integran datos de imágenes (ej. comportamiento de nado) ni análisis de agua en laboratorio.
4. **Simulación económica simplificada:** Los precios de mercado y costos se asumen constantes durante el ciclo.
5. **RL en simulación:** El agente se entrena en un entorno simulado; el despliegue en campo requiere validación adicional.

### Trabajo Futuro

1. **Aprendizaje semi-supervisado:** Combinar el autoencoder con etiquetas limitadas para reducir dependencia de datos anotados.
2. **Visión por computadora:** Integrar análisis de video para detectar cambios de comportamiento del camarón.
3. **Transfer learning:** Pre-entrenar en múltiples granjas y adaptar con fine-tuning local.
4. **Optimización multi-objetivo:** Extender la función de recompensa para optimizar simultáneamente ROI y sostenibilidad ambiental.
5. **Despliegue en edge:** Portar el modelo a dispositivos embebidos para inferencia local en tiempo real.
6. **Federated learning:** Entrenar colaborativamente entre granjas sin compartir datos sensibles.
7. **Integración de pronóstico climático:** Incorporar predicciones meteorológicas a 7 días como features del modelo.

---

*Documento de metodología v1.0 — Simulador de Bioseguridad Inteligente para Camarón*
