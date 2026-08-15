"""
Publicador de prueba MQTT — simula un nodo sensor ESP32.

Publica lecturas JSON en bioseguridad/sensors/{pond_id} cada N segundos
para probar el bucle completo:  sensor -> broker MQTT -> MqttBridge
-> WebSocket -> dashboard.

Uso:
    python scripts/mqtt_publisher_test.py --pond pond-001 --broker localhost --interval 2

Requiere: pip install paho-mqtt
"""

import argparse
import json
import random
import time

import paho.mqtt.client as mqtt

TOPIC_PREFIX = "bioseguridad/sensors/"

SENSORS = {
    "pond-001": {"ph": 7.8, "dissolved_oxygen": 5.2, "salinity": 30.1, "turbidity": 25.0, "temperature": 29.1, "ammonia": 0.35},
    "pond-002": {"ph": 7.9, "dissolved_oxygen": 5.6, "salinity": 30.5, "turbidity": 22.0, "temperature": 28.9, "ammonia": 0.28},
    "pond-003": {"ph": 7.2, "dissolved_oxygen": 3.8, "salinity": 29.8, "turbidity": 55.0, "temperature": 30.2, "ammonia": 1.20},
}


def evolve(base: dict) -> dict:
    """Random walk alrededor del valor base."""
    noise = {
        "ph": random.uniform(-0.05, 0.05),
        "dissolved_oxygen": random.uniform(-0.15, 0.15),
        "salinity": random.uniform(-0.1, 0.1),
        "turbidity": random.uniform(-1.0, 1.0),
        "temperature": random.uniform(-0.1, 0.1),
        "ammonia": random.uniform(-0.02, 0.02),
    }
    out = {k: max(0.0, base[k] + noise[k]) for k in base}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Publicador MQTT de prueba (simula ESP32)")
    parser.add_argument("--pond", default="pond-001", help="Identificador de piscina (default: pond-001)")
    parser.add_argument("--broker", default="localhost", help="Host del broker MQTT")
    parser.add_argument("--port", type=int, default=1883, help="Puerto MQTT")
    parser.add_argument("--interval", type=float, default=2.0, help="Segundos entre publicaciones")
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim-{args.pond}")
    client.connect(args.broker, args.port, 60)
    client.loop_start()

    base = SENSORS.get(args.pond, SENSORS["pond-001"])
    topic = TOPIC_PREFIX + args.pond
    print(f"Publicando en {topic} (broker {args.broker}:{args.port}) cada {args.interval}s — Ctrl+C para salir")

    try:
        while True:
            reading = evolve(base)
            payload = json.dumps(reading)
            info = client.publish(topic, payload, qos=1)
            print(f"[{time.strftime('%H:%M:%S')}] {topic} -> {payload}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nDetenido.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
