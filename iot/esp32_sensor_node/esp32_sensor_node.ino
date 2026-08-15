/*
 * Nodo Sensor ESP32 — Bioseguridad Camarón AI
 * =============================================
 * Publica lecturas de calidad de agua por MQTT al topic:
 *     bioseguridad/sensors/{POND_ID}
 *
 * Payload JSON:
 *     {"ph":7.8,"dissolved_oxygen":5.2,"salinity":30.1,
 *      "turbidity":25.0,"temperature":29.1,"ammonia":0.35}
 *
 * Hardware recomendado:
 *   - pH: módulo pH-4502C (pin analógico PH_PIN)
 *   - Temperatura: DS18B20 (OneWire, pin TEMP_PIN)
 *   - Salinidad: sonda conductividad TDS/EC (analógico SAL_PIN)
 *   - Turbidez: sensor óptico SEN0189 (analógico TURB_PIN)
 *   - Oxígeno disuelto y amonio: sondas analógicas opcionales.
 *     Sin sondas, el nodo opera en modo SIMULADO para pruebas.
 *
 * Librerías (Library Manager):
 *   - PubSubClient (knolleary)
 *   - OneWire + DallasTemperature (milesburton)
 *   - ArduinoJson (bblanchon)
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ============ CONFIGURACIÓN ============
const char *WIFI_SSID = "TU_RED_WIFI";
const char *WIFI_PASS = "TU_CONTRASEÑA";

const char *MQTT_HOST = "192.168.1.100";  // IP de tu broker (o nube: broker.hivemq.com)
const int   MQTT_PORT = 1883;
const char *MQTT_USER = "";               // vacío si allow_anonymous true
const char *MQTT_PASS = "";

const char *POND_ID = "pond-001";         // identificador de la piscina
const char *SPECIES = "vannamei";         // especie para umbrales

const unsigned long PUBLISH_INTERVAL_MS = 15000;  // publicar cada 15 s
const bool SIMULATED = true;              // true = genera lecturas simuladas
// ========================================

// Pines analógicos
#define PH_PIN   34
#define SAL_PIN  35
#define TURB_PIN 36
#define DO_PIN   39
#define AMM_PIN  32
#define TEMP_PIN 4

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);
OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

unsigned long lastPublish = 0;
float simTemp = 29.0, simPh = 7.8, simDo = 5.2, simSal = 30.0, simTurb = 25.0, simAmm = 0.3;

void setup() {
  Serial.begin(115200);
  tempSensor.begin();

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Conectando a WiFi");
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi no disponible — reiniciando en 10s");
    delay(10000);
    ESP.restart();
  }
  Serial.println("\nWiFi conectado: " + WiFi.localIP().toString());

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setKeepAlive(30);
  mqtt.setBufferSize(512);
}

void loop() {
  // Si WiFi se cae en runtime, intentar reconectar
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.reconnect();
    delay(1000);
    return;
  }
  if (!mqtt.connected()) {
    reconnectMqtt();
  }
  mqtt.loop();

  if (millis() - lastPublish >= PUBLISH_INTERVAL_MS) {
    lastPublish = millis();
    publishReading();
  }
}

void reconnectMqtt() {
  while (!mqtt.connected()) {
    Serial.print("Conectando a MQTT...");
    String clientId = String("esp32-") + String((uint32_t)ESP.getEfuseMac(), HEX);
    bool ok = mqtt.connect(clientId.c_str(), MQTT_USER, MQTT_PASS);
    if (ok) {
      Serial.println(" OK");
    } else {
      Serial.print(" falló (rc=");
      Serial.print(mqtt.state());
      Serial.println("), reintento en 5s");
      delay(5000);
    }
  }
}

// ---- Lectura de sensores ----

float readTemperature() {
  if (SIMULATED) {
    simTemp += random(-15, 16) / 100.0;
    return constrain(simTemp, 26.0, 32.0);
  }
  tempSensor.requestTemperatures();
  float t = tempSensor.getTempCByIndex(0);
  return (t == DEVICE_DISCONNECTED_C) ? NAN : t;
}

float readPh() {
  if (SIMULATED) {
    simPh += random(-8, 9) / 100.0;
    return constrain(simPh, 6.5, 8.5);
  }
  // pH-4502C: voltaje -> pH (calibrar offset)
  int raw = analogRead(PH_PIN);
  float voltage = raw * (3.3 / 4095.0);
  return 7.0 + (2.5 - voltage) / 0.18;  // pendiente típica 0.18 V/pH
}

float readDissolvedOxygen() {
  if (SIMULATED) {
    simDo += random(-12, 13) / 100.0;
    return constrain(simDo, 3.0, 7.5);
  }
  int raw = analogRead(DO_PIN);
  float voltage = raw * (3.3 / 4095.0);
  return voltage / 0.6;  // sonda DO típica ~0.6 V por mg/L (calibrar)
}

float readSalinity() {
  if (SIMULATED) {
    simSal += random(-10, 11) / 100.0;
    return constrain(simSal, 25.0, 35.0);
  }
  int raw = analogRead(SAL_PIN);
  float voltage = raw * (3.3 / 4095.0);
  return voltage * 12.0;  // conversión aproximada (calibrar con conductividad)
}

float readTurbidity() {
  if (SIMULATED) {
    simTurb += random(-20, 21) / 10.0;
    return constrain(simTurb, 5.0, 60.0);
  }
  int raw = analogRead(TURB_PIN);
  float voltage = raw * (3.3 / 4095.0);
  return mapTurbidity(voltage);
}

float mapTurbidity(float voltage) {
  // SEN0189: voltaje menor = agua más turbia (calibración típica)
  if (voltage >= 2.5) return 0.0;
  if (voltage <= 1.0) return 100.0;
  return (2.5 - voltage) / 1.5 * 100.0;
}

float readAmmonia() {
  if (SIMULATED) {
    simAmm += random(-3, 4) / 100.0;
    return constrain(simAmm, 0.05, 1.5);
  }
  int raw = analogRead(AMM_PIN);
  float voltage = raw * (3.3 / 4095.0);
  return voltage / 1.2;  // sonda amonio (calibrar)
}

// ---- Publicación ----

void publishReading() {
  float t = readTemperature();
  float p = readPh();
  float d = readDissolvedOxygen();
  float s = readSalinity();
  float tu = readTurbidity();
  float a = readAmmonia();

  if (isnan(t) || isnan(p) || isnan(d) || isnan(s) || isnan(tu) || isnan(a)) {
    Serial.println("Sensor desconectado — lectura omitida");
    return;
  }

  StaticJsonDocument<256> doc;
  doc["ph"] = p;
  doc["dissolved_oxygen"] = d;
  doc["salinity"] = s;
  doc["turbidity"] = tu;
  doc["temperature"] = t;
  doc["ammonia"] = a;
  doc["species"] = SPECIES;

  String topic = String("bioseguridad/sensors/") + POND_ID;
  char payload[256];
  size_t n = serializeJson(doc, payload, sizeof(payload));

  bool ok = mqtt.publish(topic.c_str(), payload, n);
  Serial.printf("[%lu] %s -> %s (%s)\n", millis(), topic.c_str(), payload, ok ? "OK" : "FALLÓ");
}
