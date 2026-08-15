"""
Prueba E2E: MQTT (broker público) -> MqttBridge -> WebSocket -> cliente.

Flujo:
1. Publica una lectura en test.mosquitto.org:1883 (topic bioseguridad/sensors/pond-001)
2. La API (con MQTT_ENABLED=1) recibe el mensaje via MqttBridge
3. El WebSocket /ws/pond/pond-001 reenvía la lectura al cliente

Uso (API ya corriendo con MQTT_ENABLED=1):
    python scripts/e2e_mqtt_websocket_test.py
"""

import asyncio
import json
import time

import paho.mqtt.client as mqtt
import websockets

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "bioseguridad/sensors/pond-001"
WS_URL = "ws://localhost:8000/ws/pond/pond-001"


async def main() -> int:
    # 1) Publicar por MQTT
    reading = {
        "ph": 7.61, "dissolved_oxygen": 3.42, "salinity": 29.9,
        "turbidity": 58.0, "temperature": 30.4, "ammonia": 1.55,
        "species": "vannamei",
    }
    pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="e2e-pub")
    pub.connect(BROKER, PORT, 30)
    pub.loop_start()
    try:
        info = pub.publish(TOPIC, json.dumps(reading), qos=1)
        info.wait_for_publish(timeout=10)
        print(f"[1] Publicado en {TOPIC} (mosquitto.org): {json.dumps(reading)}")

        # 2) Conectar WebSocket a la API local
        async with websockets.connect(WS_URL) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            print(f"[2] Mensaje inicial: type={first.get('type')}")

            # 3) Esperar la lectura proveniente de MQTT (source=mqtt)
            deadline = time.time() + 30
            while time.time() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") == "reading":
                    print(f"[3] Lectura recibida por WebSocket: source={msg.get('source')} pond={msg.get('pond_id')}")
                    if msg.get("source") == "mqtt":
                        print(f"    sensores: {msg['sensors']}")
                        return 0
                    print("    (aún no es la lectura MQTT, esperando...)")
        return 1
    finally:
        pub.loop_stop()
        pub.disconnect()


if __name__ == "__main__":
    code = asyncio.run(main())
    print("RESULTADO:", "E2E OK — MQTT -> bridge -> WebSocket" if code == 0 else "FALLÓ: no llegó la lectura MQTT")
    raise SystemExit(code)
