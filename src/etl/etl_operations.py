"""
ETL para datos operacionales de estanques camaroneros.

Procesa datos de manejo: densidad de siembra, tasa de alimentación,
tratamientos aplicados y otras variables operacionales.

Autor: Equipo de Bioseguridad Camarón AI
Fecha: 2026-08-10
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


class OperationsETL:
    """
    Pipeline ETL para datos operacionales de estanques.

    Carga, normaliza y fusiona datos de manejo con datos de sensores
    para enriquecer el conjunto de características del modelo.

    Attributes:
        df: DataFrame interno con los datos operacionales cargados.
    """

    df: Optional[pd.DataFrame] = None

    # ─── Carga ───────────────────────────────────────────────────────

    def load_raw(self, path: str) -> pd.DataFrame:
        """
        Carga datos operacionales crudos desde CSV o Parquet.

        Args:
            path: Ruta al archivo de datos operacionales.

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

        # Convertir fecha
        if "fecha" in self.df.columns:
            self.df["fecha"] = pd.to_datetime(self.df["fecha"])

        print(f"[OperationsETL] Cargados {len(self.df)} registros operacionales desde {path}")
        return self.df

    # ─── Normalización ───────────────────────────────────────────────

    def normalize_operations(self) -> pd.DataFrame:
        """
        Normaliza los datos operacionales.

        - Estandariza nombres de columnas.
        - Convierte tipos numéricos.
        - Valida rangos de densidad y alimentación.
        - Rellena valores faltantes con interpolación.

        Returns:
            DataFrame normalizado.

        Raises:
            ValueError: Si no hay datos cargados.
        """
        if self.df is None:
            raise ValueError("No hay datos cargados. Ejecute load_raw() primero.")

        df = self.df.copy()

        # Asegurar columnas obligatorias
        required = ["estanque_id", "fecha"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Columna obligatoria faltante: {col}")

        # Asegurar tipos numéricos
        numeric_cols = [
            "densidad_siembra",
            "tasa_alimentacion",
            "peso_promedio",
            "edad_dias",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Validar y corregir rangos
        if "densidad_siembra" in df.columns:
            df.loc[df["densidad_siembra"] < 0, "densidad_siembra"] = np.nan
            df["densidad_siembra"] = df["densidad_siembra"].interpolate(method="linear")

        if "tasa_alimentacion" in df.columns:
            df.loc[df["tasa_alimentacion"] < 0, "tasa_alimentacion"] = np.nan
            df["tasa_alimentacion"] = df["tasa_alimentacion"].interpolate(method="linear")

        # Normalizar tratamiento aplicado
        if "tratamiento_aplicado" in df.columns:
            df["tratamiento_aplicado"] = df["tratamiento_aplicado"].fillna("Ninguno").str.strip()

        # Normalizar tipo_alimento
        if "tipo_alimento" in df.columns:
            df["tipo_alimento"] = df["tipo_alimento"].fillna("Estándar").str.strip()

        # Ordenar
        df = df.sort_values(["estanque_id", "fecha"]).reset_index(drop=True)

        self.df = df
        print(f"[OperationsETL] Datos normalizados: {len(df)} registros")
        return df

    # ─── Fusión con sensores ─────────────────────────────────────────

    def merge_with_sensors(self, sensor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fusiona datos operacionales con datos de sensores.

        Realiza un merge por ``estanque_id`` usando ``merge_asof`` para
        alinear cada registro de sensor con la operación más reciente
        en ese estanque (forward fill de operaciones).

        Args:
            sensor_df: DataFrame de sensores con columnas ``estanque_id``
                y ``timestamp``.

        Returns:
            DataFrame fusionado con columnas operacionales añadidas.

        Raises:
            ValueError: Si no hay datos operacionales cargados.
        """
        if self.df is None:
            raise ValueError("No hay datos operacionales. Ejecute load_raw() y normalize_operations() primero.")

        sensor = sensor_df.copy()
        ops = self.df.copy()

        # Asegurar timestamps
        sensor["timestamp"] = pd.to_datetime(sensor["timestamp"])
        if "fecha" in ops.columns:
            ops["timestamp"] = ops["fecha"]
        else:
            raise ValueError("Los datos operacionales deben tener columna 'fecha'")

        # Ordenar por timestamp para merge_asof
        sensor = sensor.sort_values(["estanque_id", "timestamp"])
        ops = ops.sort_values(["estanque_id", "timestamp"])

        merged_parts: list[pd.DataFrame] = []

        for pond_id in sensor["estanque_id"].unique():
            s = sensor[sensor["estanque_id"] == pond_id].copy()
            o = ops[ops["estanque_id"] == pond_id].copy()

            if len(o) == 0:
                # Sin operaciones para este estanque: rellenar con defaults
                for col in ["densidad_siembra", "tasa_alimentacion", "peso_promedio", "edad_dias"]:
                    if col in ops.columns:
                        s[col] = np.nan
                if "tratamiento_aplicado" in ops.columns:
                    s["tratamiento_aplicado"] = "Ninguno"
                if "tipo_alimento" in ops.columns:
                    s["tipo_alimento"] = "Estándar"
            else:
                s = pd.merge_asof(
                    s,
                    o[
                        ["timestamp"]
                        + [c for c in o.columns if c not in ["estanque_id", "fecha", "timestamp"]]
                    ],
                    on="timestamp",
                    direction="backward",
                )

            merged_parts.append(s)

        result = pd.concat(merged_parts, ignore_index=True)

        # Llenar NaN restantes con forward fill por estanque
        ops_cols = [c for c in result.columns if c not in sensor_df.columns]
        for col in ops_cols:
            if result[col].dtype in [np.float64, np.int64, float, int]:
                result[col] = result.groupby("estanque_id")[col].ffill()
            else:
                result[col] = result.groupby("estanque_id")[col].ffill()

        print(f"[OperationsETL] Fusión completada: {len(result)} registros, +{len(ops_cols)} columnas operacionales")
        return result

    # ─── Generación de datos sintéticos ──────────────────────────────

    @staticmethod
    def generate_synthetic_operations(
        n_ponds: int = 5,
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Genera datos operacionales sintéticos para testing.

        Simula registros diarios por estanque con densidad de siembra,
        tasa de alimentación, peso promedio, edad, tipo de alimento
        y tratamientos aplicados.

        Args:
            n_ponds: Número de estanques.
            days: Número de días del período.

        Returns:
            DataFrame con datos operacionales sintéticos.
        """
        np.random.seed(456)
        start_date = datetime(2024, 1, 1)
        dates = pd.date_range(start=start_date, periods=days, freq="D")

        records: list[dict] = []

        for pond_id in range(1, n_ponds + 1):
            # Parámetros iniciales del ciclo de cultivo
            densidad_inicial = np.random.uniform(15, 45)  # organismos/m²
            peso_inicial = np.random.uniform(0.5, 2.0)    # gramos
            edad_inicial = np.random.randint(10, 30)       # días

            for i, fecha in enumerate(dates):
                # La densidad disminuye ligeramente por mortalidad natural
                densidad = densidad_inicial * (1 - 0.001 * i) * np.random.uniform(0.99, 1.01)
                # El peso aumenta con el tiempo (curva logarítmica)
                peso = peso_inicial + 8.0 * np.log(1 + i / 10.0) + np.random.normal(0, 0.3)
                # Tasa de alimentación ajustada al peso
                tasa = peso * densidad * 0.03 * np.random.uniform(0.9, 1.1)
                edad = edad_inicial + i

                # Tratamiento aleatorio (~10% de los días)
                if np.random.random() < 0.1:
                    tratamiento = np.random.choice(
                        ["Probiótico", "Antibiótico", "Cambio de agua", "Cal", "Zeolita"]
                    )
                else:
                    tratamiento = "Ninguno"

                tipo_alimento = np.random.choice(
                    ["Estándar", "Alta proteína", "Pre-iniciador", "Engorde"],
                    p=[0.4, 0.25, 0.15, 0.2],
                )

                records.append({
                    "estanque_id": f"E{pond_id:03d}",
                    "fecha": fecha,
                    "densidad_siembra": round(densidad, 1),
                    "tasa_alimentacion": round(tasa, 2),
                    "peso_promedio": round(peso, 2),
                    "edad_dias": int(edad),
                    "tipo_alimento": tipo_alimento,
                    "tratamiento_aplicado": tratamiento,
                })

        df = pd.DataFrame(records)
        print(f"[OperationsETL] Generados {len(df)} registros operacionales ({n_ponds} estanques × {days} días)")
        return df


# ─── Punto de entrada ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ETL Operaciones – Bioseguridad Camarón AI")
    print("=" * 60)

    etl = OperationsETL()

    # Generar datos sintéticos
    print("\n→ Generando datos operacionales sintéticos...")
    synthetic = OperationsETL.generate_synthetic_operations(n_ponds=5, days=90)
    etl.df = synthetic

    # Guardar datos crudos
    raw_path = ".cluster/bioseguridad-camaron-AI/repo/data/raw/operaciones_crudas.csv"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    synthetic.to_csv(raw_path, index=False)
    print(f"   Datos crudos guardados en: {raw_path}")

    # Cargar y normalizar
    print("\n→ Cargando y normalizando...")
    etl.load_raw(raw_path)
    etl.normalize_operations()

    # Generar sensores para probar fusión
    print("\n→ Generando sensores para fusión...")
    from src.etl.etl_sensors import SensorETL
    sensor_df = SensorETL.generate_synthetic_data(n_ponds=5, days=90)

    # Fusionar
    print("\n→ Fusionando con sensores...")
    merged = etl.merge_with_sensors(sensor_df)

    # Exportar
    print("\n→ Exportando a Parquet...")
    output = ".cluster/bioseguridad-camaron-AI/repo/data/processed/operaciones.parquet"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    merged.to_parquet(output, index=False, engine="pyarrow")

    print(f"\n✓ Pipeline de operaciones completado. Output: {output}")
    print(f"  Columnas finales: {list(merged.columns)}")
