"""
ETL para datos climáticos aplicados a estanques camaroneros.

Procesa variables meteorológicas: temperatura ambiente, humedad relativa,
precipitación y velocidad del viento. Resamplea y fusiona con datos
de sensores para enriquecer el modelo predictivo.

Autor: Equipo de Bioseguridad Camarón AI
Fecha: 2026-08-10
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


class ClimateETL:
    """
    Pipeline ETL para datos climáticos.

    Carga, resamplea y fusiona datos meteorológicos con datos de
    sensores de estanques para proporcionar contexto ambiental.

    Attributes:
        df: DataFrame interno con los datos climáticos cargados.
    """

    df: Optional[pd.DataFrame] = None

    # ─── Carga ───────────────────────────────────────────────────────

    def load_raw(self, path: str) -> pd.DataFrame:
        """
        Carga datos climáticos crudos desde CSV o Parquet.

        Args:
            path: Ruta al archivo de datos climáticos.

        Returns:
            DataFrame con los datos cargados.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si la extensión no es soportada.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            self.df = pd.read_csv(path)
        elif ext == ".parquet":
            self.df = pd.read_parquet(path)
        else:
            raise ValueError(f"Extensión no soportada: {ext}. Use .csv o .parquet")

        # Convertir timestamp
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # Asegurar tipos numéricos
        numeric_cols = [
            "temp_ambiente",
            "humedad_relativa",
            "precipitacion",
            "velocidad_viento",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        print(f"[ClimateETL] Cargados {len(self.df)} registros climáticos desde {path}")
        return self.df

    # ─── Resampleo ───────────────────────────────────────────────────

    def resample_15min(self) -> pd.DataFrame:
        """
        Remuestrea los datos climáticos a intervalos de 15 minutos.

        Agrupa por estanque y usa interpolación lineal para mantener
        continuidad en las series temporales.

        Returns:
            DataFrame resampleado a 15 minutos.

        Raises:
            ValueError: Si no hay datos cargados.
        """
        if self.df is None:
            raise ValueError("No hay datos cargados. Ejecute load_raw() primero.")

        df = self.df.copy()
        df = df.sort_values(["estanque_id", "timestamp"])

        resampled_parts: list[pd.DataFrame] = []

        for pond_id, group in df.groupby("estanque_id"):
            group = group.set_index("timestamp")
            # Separar columnas no numéricas antes del resampleo
            numeric_cols = group.select_dtypes(include=[np.number]).columns.tolist()
            group_numeric = group[numeric_cols]
            group_numeric = group_numeric.resample("15min").mean()
            group_numeric = group_numeric.interpolate(method="linear", limit=8)  # máx 2h de interpolación
            group_numeric = group_numeric.reset_index()
            group_numeric["estanque_id"] = pond_id
            resampled_parts.append(group_numeric)

        self.df = pd.concat(resampled_parts, ignore_index=True)
        print(f"[ClimateETL] Resampleado a 15 min: {len(self.df)} registros")
        return self.df

    # ─── Fusión con sensores ─────────────────────────────────────────

    def merge_with_sensors(self, sensor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fusiona datos climáticos con datos de sensores.

        Usa ``merge_asof`` por ``estanque_id`` para alinear cada lectura
        de sensor con el dato climático más cercano en el tiempo.

        Args:
            sensor_df: DataFrame de sensores con columnas ``estanque_id``
                y ``timestamp``.

        Returns:
            DataFrame fusionado con columnas climáticas añadidas.

        Raises:
            ValueError: Si no hay datos climáticos cargados.
        """
        if self.df is None:
            raise ValueError("No hay datos climáticos. Ejecute load_raw() y resample_15min() primero.")

        sensor = sensor_df.copy()
        climate = self.df.copy()

        sensor["timestamp"] = pd.to_datetime(sensor["timestamp"])
        climate["timestamp"] = pd.to_datetime(climate["timestamp"])

        sensor = sensor.sort_values(["estanque_id", "timestamp"])
        climate = climate.sort_values(["estanque_id", "timestamp"])

        merged_parts: list[pd.DataFrame] = []

        climate_cols = [
            c for c in climate.columns
            if c not in ["estanque_id", "timestamp"]
        ]

        for pond_id in sensor["estanque_id"].unique():
            s = sensor[sensor["estanque_id"] == pond_id].copy()
            c = climate[climate["estanque_id"] == pond_id].copy()

            if len(c) == 0:
                # Sin datos climáticos para este estanque: rellenar con NaN
                for col in climate_cols:
                    s[col] = np.nan
            else:
                s = pd.merge_asof(
                    s,
                    c[["timestamp"] + climate_cols],
                    on="timestamp",
                    direction="nearest",
                )

            merged_parts.append(s)

        result = pd.concat(merged_parts, ignore_index=True)

        # Forward fill para cualquier NaN restante (por estanque)
        for col in climate_cols:
            result[col] = result.groupby("estanque_id")[col].ffill()
            result[col] = result.groupby("estanque_id")[col].bfill()

        print(f"[ClimateETL] Fusión completada: {len(result)} registros, +{len(climate_cols)} columnas climáticas")
        return result

    # ─── Generación de datos sintéticos ──────────────────────────────

    @staticmethod
    def generate_synthetic_climate(
        n_ponds: int = 5,
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Genera datos climáticos sintéticos para testing.

        Simula variables meteorológicas con ciclos diurnos, patrones
        estacionales y eventos de lluvia aleatorios.

        Args:
            n_ponds: Número de estanques (cada uno recibe datos climáticos
                ligeramente distintos para simular microclimas).
            days: Número de días del período.

        Returns:
            DataFrame con datos climáticos sintéticos.
        """
        np.random.seed(789)
        start_date = datetime(2024, 1, 1)
        # Datos horarios (más frecuentes que sensores para permitir resampleo)
        n_records_per_pond = days * 24
        timestamps = pd.date_range(
            start=start_date,
            periods=n_records_per_pond,
            freq="1h",
        )

        all_data: list[pd.DataFrame] = []

        for pond_id in range(1, n_ponds + 1):
            # Parámetros base con variación por microclima
            base_temp = 27.0 + np.random.uniform(-1.5, 1.5)
            base_humidity = 75.0 + np.random.uniform(-5.0, 5.0)
            base_wind = 8.0 + np.random.uniform(-2.0, 3.0)

            hours = np.arange(n_records_per_pond)

            # Ciclo diurno
            day_cycle = np.sin(2 * np.pi * hours / 24.0)
            # Estacional (siempre mismo año)
            seasonal = np.sin(2 * np.pi * hours / (days * 24))

            # Temperatura ambiente: sigue ciclo diurno + estacional
            temp_amb = base_temp + 5.0 * day_cycle + 1.5 * seasonal + np.random.normal(0, 1.0, n_records_per_pond)

            # Humedad relativa: inversa al ciclo diurno
            humidity = base_humidity - 10.0 * day_cycle + 3.0 * seasonal + np.random.normal(0, 3.0, n_records_per_pond)
            humidity = np.clip(humidity, 30.0, 100.0)

            # Velocidad del viento: más alta durante el día
            wind = base_wind + 3.0 * np.maximum(day_cycle, 0) + np.random.normal(0, 1.5, n_records_per_pond)
            wind = np.clip(wind, 0.0, 40.0)

            # Precipitación: eventos aleatorios (distribución de Poisson)
            # ~15% de las horas tienen lluvia
            rain_mask = np.random.random(n_records_per_pond) < 0.15
            precip = np.zeros(n_records_per_pond)
            precip[rain_mask] = np.random.exponential(5.0, rain_mask.sum())
            # Amplificar lluvia en temporada lluviosa (estacional)
            wet_season = np.maximum(seasonal, 0)
            precip = precip * (1.0 + 2.0 * wet_season)

            pond_df = pd.DataFrame({
                "estanque_id": f"E{pond_id:03d}",
                "timestamp": timestamps,
                "temp_ambiente": np.round(temp_amb, 1),
                "humedad_relativa": np.round(humidity, 1),
                "precipitacion": np.round(precip, 1),
                "velocidad_viento": np.round(wind, 1),
            })
            all_data.append(pond_df)

        result = pd.concat(all_data, ignore_index=True)
        print(f"[ClimateETL] Generados {len(result)} registros climáticos ({n_ponds} estanques × {days} días)")
        return result


# ─── Punto de entrada ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ETL Clima – Bioseguridad Camarón AI")
    print("=" * 60)

    etl = ClimateETL()

    # Generar datos sintéticos
    print("\n→ Generando datos climáticos sintéticos...")
    synthetic = ClimateETL.generate_synthetic_climate(n_ponds=5, days=90)
    etl.df = synthetic

    # Guardar datos crudos
    raw_path = ".cluster/bioseguridad-camaron-AI/repo/data/raw/clima_crudo.csv"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    synthetic.to_csv(raw_path, index=False)
    print(f"   Datos crudos guardados en: {raw_path}")

    # Cargar
    print("\n→ Cargando datos climáticos...")
    etl.load_raw(raw_path)

    # Resamplear
    print("\n→ Resampleando a 15 minutos...")
    etl.resample_15min()

    # Generar sensores para probar fusión
    print("\n→ Generando sensores para fusión...")
    from src.etl.etl_sensors import SensorETL
    sensor_df = SensorETL.generate_synthetic_data(n_ponds=5, days=90)

    # Fusionar
    print("\n→ Fusionando con sensores...")
    merged = etl.merge_with_sensors(sensor_df)

    # Exportar
    print("\n→ Exportando a Parquet...")
    output = ".cluster/bioseguridad-camaron-AI/repo/data/processed/clima.parquet"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    merged.to_parquet(output, index=False, engine="pyarrow")

    print(f"\n✓ Pipeline climático completado. Output: {output}")
    print(f"  Columnas finales: {list(merged.columns)}")
