"""
ETL para datos de sensores de estanques camaroneros.

Procesa lecturas de: pH, oxígeno disuelto, salinidad, turbidez,
temperatura y amoníaco. Incluye detección de fallos de sensor,
resampleo temporal y exportación a Parquet.

Autor: Equipo de Bioseguridad Camarón AI
Fecha: 2026-08-10
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ─── Rangos físicamente válidos para cada variable ───
SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "ph": (0.0, 14.0),
    "oxigeno_disuelto": (0.0, 20.0),       # mg/L
    "temperatura": (10.0, 40.0),            # °C
    "salinidad": (0.0, 45.0),               # ppt
    "turbidez": (0.0, 1000.0),              # NTU
    "amonio": (0.0, 10.0),                  # mg/L
}

# Variables numéricas que esperamos en el DataFrame
NUMERIC_COLUMNS: list[str] = list(SENSOR_RANGES.keys())

# Columnas de identificación
ID_COLUMNS: list[str] = ["estanque_id", "timestamp"]


@dataclass
class SensorETL:
    """
    Pipeline ETL para datos de sensores de estanques.

    Procesa datos crudos de sensores multivariables, detecta fallos,
    resamplea a intervalos regulares de 15 minutos y exporta a Parquet.

    Attributes:
        df: DataFrame interno con los datos crudos cargados.
        failures: DataFrame con los registros marcados como fallos.
    """

    df: Optional[pd.DataFrame] = None
    failures: Optional[pd.DataFrame] = None

    # ─── Carga ───────────────────────────────────────────────────────

    def load_raw(self, path: str) -> pd.DataFrame:
        """
        Carga datos crudos desde un archivo CSV o Parquet.

        Args:
            path: Ruta al archivo de datos crudos (.csv o .parquet).

        Returns:
            DataFrame con los datos crudos cargados.

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

        # Asegurar que timestamp sea datetime
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # Asegurar tipos numéricos
        for col in NUMERIC_COLUMNS:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        print(f"[SensorETL] Cargados {len(self.df)} registros desde {path}")
        return self.df

    # ─── Resampleo ───────────────────────────────────────────────────

    def resample_15min(self) -> pd.DataFrame:
        """
        Remuestrea los datos a intervalos regulares de 15 minutos.

        Agrupa por estanque y resamplea usando interpolación lineal para
        mantener continuidad en las series temporales.

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
            group_numeric = group_numeric.interpolate(method="linear", limit=4)  # máx 1h de interpolación
            group_numeric = group_numeric.reset_index()
            group_numeric["estanque_id"] = pond_id
            resampled_parts.append(group_numeric)

        self.df = pd.concat(resampled_parts, ignore_index=True)
        print(f"[SensorETL] Resampleado a 15 min: {len(self.df)} registros")
        return self.df

    # ─── Detección de fallos ─────────────────────────────────────────

    def detect_sensor_failures(self) -> pd.DataFrame:
        """
        Detecta fallos en los sensores usando tres criterios:

        1. **Valores fuera de rango físico**: Lecturas imposibles
           (ej. pH < 0 o pH > 14).
        2. **Sensor atascado**: Lecturas idénticas consecutivas
           (más de 3 lecturas iguales seguidas).
        3. **Huecos temporales**: Brechas mayores a 1 hora entre lecturas.

        Returns:
            DataFrame con los registros marcados como fallos, incluyendo
            la columna ``fallo_tipo`` con el tipo de fallo detectado.
        """
        if self.df is None:
            raise ValueError("No hay datos cargados. Ejecute load_raw() primero.")

        df = self.df.copy()
        failure_records: list[dict] = []

        # ── 1. Valores fuera de rango físico ──
        for col, (min_val, max_val) in SENSOR_RANGES.items():
            if col not in df.columns:
                continue
            mask = (df[col] < min_val) | (df[col] > max_val)
            for idx in df[mask].index:
                failure_records.append({
                    "estanque_id": df.loc[idx, "estanque_id"],
                    "timestamp": df.loc[idx, "timestamp"],
                    "variable": col,
                    "valor": df.loc[idx, col],
                    "fallo_tipo": "fuera_de_rango",
                    "detalle": f"Valor {df.loc[idx, col]:.2f} fuera de rango [{min_val}, {max_val}]",
                })

        # ── 2. Sensor atascado (lecturas consecutivas idénticas) ──
        for pond_id, group in df.groupby("estanque_id"):
            group = group.sort_values("timestamp")
            for col in NUMERIC_COLUMNS:
                if col not in group.columns:
                    continue
                vals = group[col].values
                # Detectar secuencias de valores idénticos consecutivos
                if len(vals) < 4:
                    continue
                # Crear mascara de cambios
                changes = np.diff(vals)
                # Posiciones donde el valor no cambia
                same = changes == 0
                # Contar rachas de valores iguales
                streak_start = None
                streak_count = 0
                for i, s in enumerate(same):
                    if s:
                        if streak_start is None:
                            streak_start = i
                        streak_count += 1
                    else:
                        if streak_count >= 3 and streak_start is not None:
                            # Marcar los registros atascados
                            for j in range(streak_start, streak_start + streak_count + 1):
                                failure_records.append({
                                    "estanque_id": pond_id,
                                    "timestamp": group.iloc[j]["timestamp"],
                                    "variable": col,
                                    "valor": vals[j],
                                    "fallo_tipo": "sensor_atascado",
                                    "detalle": f"{streak_count + 1} lecturas consecutivas idénticas: {vals[j]:.2f}",
                                })
                        streak_start = None
                        streak_count = 0
                # Verificar al final del bucle
                if streak_count >= 3 and streak_start is not None:
                    for j in range(streak_start, streak_start + streak_count + 1):
                        failure_records.append({
                            "estanque_id": pond_id,
                            "timestamp": group.iloc[j]["timestamp"],
                            "variable": col,
                            "valor": vals[j],
                            "fallo_tipo": "sensor_atascado",
                            "detalle": f"{streak_count + 1} lecturas consecutivas idénticas: {vals[j]:.2f}",
                        })

        # ── 3. Huecos temporales > 1 hora ──
        for pond_id, group in df.groupby("estanque_id"):
            group = group.sort_values("timestamp")
            timestamps = group["timestamp"].values
            for i in range(1, len(timestamps)):
                gap = (timestamps[i] - timestamps[i - 1])
                gap_minutes = gap.astype("timedelta64[m]").astype(int)
                if gap_minutes > 60:
                    failure_records.append({
                        "estanque_id": pond_id,
                        "timestamp": pd.Timestamp(timestamps[i]),
                        "variable": "timestamp",
                        "valor": float(gap_minutes),
                        "fallo_tipo": "hueco_temporal",
                        "detalle": f"Hueco de {gap_minutes} minutos entre lecturas",
                    })

        self.failures = pd.DataFrame(failure_records)
        if len(self.failures) > 0:
            print(f"[SensorETL] Detectados {len(self.failures)} fallos de sensor")
        else:
            print("[SensorETL] No se detectaron fallos")
            self.failures = pd.DataFrame(
                columns=["estanque_id", "timestamp", "variable", "valor", "fallo_tipo", "detalle"]
            )
        return self.failures

    # ─── Sincronización de timestamps ────────────────────────────────

    def sync_timestamps(self) -> pd.DataFrame:
        """
        Sincroniza los timestamps entre estanques para asegurar alineación temporal.

        Crea un índice temporal común de 15 minutos y reindexa cada estanque
        para que todos compartan los mismos puntos temporales.

        Returns:
            DataFrame con timestamps sincronizados.
        """
        if self.df is None:
            raise ValueError("No hay datos cargados. Ejecute load_raw() primero.")

        df = self.df.copy()
        df = df.sort_values(["estanque_id", "timestamp"])

        # Crear índice temporal común
        global_start = df["timestamp"].min().floor("15min")
        global_end = df["timestamp"].max().ceil("15min")
        common_index = pd.date_range(start=global_start, end=global_end, freq="15min")

        synced_parts: list[pd.DataFrame] = []
        for pond_id, group in df.groupby("estanque_id"):
            group = group.set_index("timestamp")
            # Seleccionar solo columnas numéricas para evitar errores con object dtype
            numeric_cols = group.select_dtypes(include=[np.number]).columns.tolist()
            group_numeric = group[numeric_cols]
            group_numeric = group_numeric.reindex(common_index)
            group_numeric.index.name = "timestamp"
            group_numeric = group_numeric.interpolate(method="linear", limit=4)
            group_numeric = group_numeric.reset_index()
            group_numeric["estanque_id"] = pond_id
            synced_parts.append(group_numeric)

        self.df = pd.concat(synced_parts, ignore_index=True)
        print(f"[SensorETL] Timestamps sincronizados: {len(self.df)} registros")
        return self.df

    # ─── Exportación ─────────────────────────────────────────────────

    def to_parquet(self, output_path: str) -> str:
        """
        Exporta los datos procesados a formato Parquet.

        Args:
            output_path: Ruta de salida para el archivo .parquet.

        Returns:
            Ruta del archivo escrito.
        """
        if self.df is None:
            raise ValueError("No hay datos para exportar. Ejecute el pipeline primero.")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.df.to_parquet(output_path, index=False, engine="pyarrow")
        print(f"[SensorETL] Datos exportados a {output_path}")
        return output_path

    # ─── Generación de datos sintéticos ──────────────────────────────

    @staticmethod
    def generate_synthetic_data(
        n_ponds: int = 5,
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Genera datos sintéticos realistas de sensores para testing.

        Crea datos con ciclos diarios, ruido gaussiano y variaciones
        estacionales para simular condiciones de estanques camaroneros.

        Args:
            n_ponds: Número de estanques a simular.
            days: Número de días de datos a generar.

        Returns:
            DataFrame con datos sintéticos de sensores.
        """
        np.random.seed(42)
        start_date = datetime(2024, 1, 1)
        # Intervalo de 15 minutos
        n_records_per_pond = days * 24 * 4  # 4 lecturas por hora
        timestamps = pd.date_range(
            start=start_date,
            periods=n_records_per_pond,
            freq="15min",
        )

        all_data: list[pd.DataFrame] = []

        for pond_id in range(1, n_ponds + 1):
            # Parámetros base por estanque (con algo de variación)
            base_ph = 7.5 + np.random.uniform(-0.5, 0.5)
            base_do = 6.0 + np.random.uniform(-1.0, 1.5)
            base_temp = 28.0 + np.random.uniform(-2.0, 2.0)
            base_sal = 25.0 + np.random.uniform(-5.0, 5.0)
            base_turb = 50.0 + np.random.uniform(-20.0, 30.0)
            base_amm = 0.5 + np.random.uniform(-0.2, 0.3)

            # Ciclo diurno (seno con período 24h)
            hours = np.arange(n_records_per_pond) * 0.25  # cada 15 min = 0.25 h
            day_cycle = np.sin(2 * np.pi * hours / 24.0)

            # Tendencia estacional lenta
            seasonal = np.sin(2 * np.pi * np.arange(n_records_per_pond) / (days * 4))

            # Generar variables
            ph = base_ph + 0.3 * day_cycle + np.random.normal(0, 0.15, n_records_per_pond)
            do = base_do + 1.5 * day_cycle + np.random.normal(0, 0.4, n_records_per_pond)
            temp = base_temp + 1.5 * day_cycle + 0.5 * seasonal + np.random.normal(0, 0.3, n_records_per_pond)
            sal = base_sal + 2.0 * seasonal + np.random.normal(0, 0.8, n_records_per_pond)
            turb = base_turb + 15.0 * seasonal + np.random.normal(0, 5.0, n_records_per_pond)
            amm = base_amm + 0.15 * seasonal + np.random.normal(0, 0.05, n_records_per_pond)

            # Asegurar valores dentro de rango físico
            ph = np.clip(ph, 5.5, 9.5)
            do = np.clip(do, 2.0, 12.0)
            temp = np.clip(temp, 20.0, 35.0)
            sal = np.clip(sal, 10.0, 40.0)
            turb = np.clip(turb, 5.0, 200.0)
            amm = np.clip(amm, 0.01, 3.0)

            # Inyectar algunos fallos (~0.5% de los registros)
            n_faults = int(n_records_per_pond * 0.005)
            if n_faults > 0:
                fault_idx = np.random.choice(n_records_per_pond, n_faults, replace=False)
                # Valores fuera de rango en pH
                n_ph_faults = n_faults // 3
                ph[fault_idx[:n_ph_faults]] = np.random.choice([-1.0, 15.0], n_ph_faults)
                # Valores fuera de rango en oxígeno disuelto
                n_do_faults = n_faults // 3
                do_slice = fault_idx[n_ph_faults : n_ph_faults + n_do_faults]
                do[do_slice] = np.random.choice([-2.0, 25.0], n_do_faults)

            pond_df = pd.DataFrame({
                "estanque_id": f"E{pond_id:03d}",
                "timestamp": timestamps,
                "ph": ph,
                "oxigeno_disuelto": do,
                "temperatura": temp,
                "salinidad": sal,
                "turbidez": turb,
                "amonio": amm,
            })
            all_data.append(pond_df)

        result = pd.concat(all_data, ignore_index=True)
        print(f"[SensorETL] Generados {len(result)} registros sintéticos ({n_ponds} estanques × {days} días)")
        return result


# ─── Punto de entrada ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ETL Sensores – Bioseguridad Camarón AI")
    print("=" * 60)

    etl = SensorETL()

    # Generar datos sintéticos
    print("\n→ Generando datos sintéticos...")
    synthetic = SensorETL.generate_synthetic_data(n_ponds=5, days=90)
    etl.df = synthetic

    # Guardar datos crudos temporales
    raw_path = ".cluster/bioseguridad-camaron-AI/repo/data/raw/sensores_crudos.csv"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    synthetic.to_csv(raw_path, index=False)
    print(f"   Datos crudos guardados en: {raw_path}")

    # Cargar desde archivo (demostración del pipeline)
    print("\n→ Cargando datos desde archivo...")
    etl.load_raw(raw_path)

    # Detectar fallos
    print("\n→ Detectando fallos de sensor...")
    failures = etl.detect_sensor_failures()
    if len(failures) > 0:
        print(f"   Fallos encontrados:\n{failures['fallo_tipo'].value_counts().to_string()}")

    # Resamplear
    print("\n→ Resampleando a 15 minutos...")
    etl.resample_15min()

    # Sincronizar timestamps
    print("\n→ Sincronizando timestamps...")
    etl.sync_timestamps()

    # Exportar
    print("\n→ Exportando a Parquet...")
    output = ".cluster/bioseguridad-camaron-AI/repo/data/processed/sensores.parquet"
    etl.to_parquet(output)

    print("\n✓ Pipeline de sensores completado.")
