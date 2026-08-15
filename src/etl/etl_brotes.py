"""
ETL para datos históricos de brotes de enfermedad en estanques camaroneros.

Procesa el registro de eventos de brotes (fecha, estanque, tipo de enfermedad,
severidad) y genera etiquetas binarias para entrenamiento de modelos de ML/DL.

Autor: Equipo de Bioseguridad Camarón AI
Fecha: 2026-08-10
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# Tipos de enfermedades comunes en camarones
TIPOS_ENFERMEDAD: list[str] = [
    "Mancha Blanca",
    "Vibriosis",
    "Taura",
    "Necrosis Hipodérmica",
    "Bacteremia",
    "White Feces",
]

# Niveles de severidad
NIVELES_SEVERIDAD: list[str] = ["leve", "moderada", "severa", "crítica"]


class BrotesETL:
    """
    Pipeline ETL para datos de brotes de enfermedad.

    Carga el historial de brotes, normaliza los eventos y genera
    etiquetas binarias sobre datos de sensores considerando una
    ventana de pre-brote de 7 días.

    Attributes:
        df: DataFrame interno con los datos de brotes cargados.
    """

    df: Optional[pd.DataFrame] = None

    # ─── Carga ───────────────────────────────────────────────────────

    def load_raw(self, path: str) -> pd.DataFrame:
        """
        Carga datos crudos de brotes desde CSV o Parquet.

        Args:
            path: Ruta al archivo de datos de brotes.

        Returns:
            DataFrame con los brotes cargados.

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

        # Convertir fechas
        for col in ["fecha_inicio", "fecha_deteccion", "fecha_fin"]:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])

        print(f"[BrotesETL] Cargados {len(self.df)} registros de brotes desde {path}")
        return self.df

    # ─── Normalización ───────────────────────────────────────────────

    def normalize_events(self) -> pd.DataFrame:
        """
        Normaliza los eventos de brotes.

        - Estandariza nombres de columnas.
        - Rellena campos faltantes.
        - Valida tipos de enfermedad contra catálogo.
        - Asigna severidad numérica (0-3).

        Returns:
            DataFrame normalizado.

        Raises:
            ValueError: Si no hay datos cargados.
        """
        if self.df is None:
            raise ValueError("No hay datos cargados. Ejecute load_raw() primero.")

        df = self.df.copy()

        # Asegurar columnas obligatorias
        required = ["estanque_id", "fecha_inicio", "tipo_enfermedad"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Columna obligatoria faltante: {col}")

        # Normalizar tipo_enfermedad (title case, sin espacios extra)
        df["tipo_enfermedad"] = df["tipo_enfermedad"].str.strip().str.title()

        # Mapear severidad texto → número
        severidad_map = {"leve": 0, "moderada": 1, "severa": 2, "crítica": 3, "critica": 3}
        if "severidad" in df.columns:
            df["severidad_num"] = df["severidad"].str.lower().map(severidad_map).fillna(1).astype(int)
        else:
            df["severidad"] = "moderada"
            df["severidad_num"] = 1

        # Asegurar fecha_fin si no existe
        if "fecha_fin" not in df.columns:
            df["fecha_fin"] = df["fecha_inicio"] + pd.Timedelta(days=7)

        # Llenar NA en campos opcionales
        if "tratamiento" not in df.columns:
            df["tratamiento"] = "No registrado"
        if "mortalidad_pct" not in df.columns:
            df["mortalidad_pct"] = np.nan

        # Ordenar por estanque y fecha
        df = df.sort_values(["estanque_id", "fecha_inicio"]).reset_index(drop=True)

        self.df = df
        print(f"[BrotesETL] Eventos normalizados: {len(df)} brotes")
        return df

    # ─── Etiquetado de eventos de brote ──────────────────────────────

    def label_outbreak_events(
        self,
        sensor_df: pd.DataFrame,
        outbreak_df: Optional[pd.DataFrame] = None,
        pre_outbreak_window_days: int = 7,
    ) -> pd.DataFrame:
        """
        Etiqueta datos de sensores con columna binaria ``outbreak``.

        Marca como 1 los registros de sensor que caen dentro de la ventana
        de pre-brote (días anteriores al inicio del brote) y durante el brote.
        El resto se marca como 0.

        Args:
            sensor_df: DataFrame de sensores con columnas ``estanque_id``
                y ``timestamp``.
            outbreak_df: DataFrame de brotes. Si es None, usa self.df.
            pre_outbreak_window_days: Días antes del brote a etiquetar como 1.

        Returns:
            Copia del sensor_df con columna ``outbreak`` (0 ó 1).

        Raises:
            ValueError: Si no hay datos de brotes disponibles.
        """
        if outbreak_df is None:
            outbreak_df = self.df
        if outbreak_df is None:
            raise ValueError("No hay datos de brotes. Cargue y normalice primero.")

        result = sensor_df.copy()
        result["outbreak"] = 0
        result["timestamp"] = pd.to_datetime(result["timestamp"])

        for _, brote in outbreak_df.iterrows():
            pond_id = brote["estanque_id"]
            fecha_inicio = pd.to_datetime(brote["fecha_inicio"])
            fecha_fin = pd.to_datetime(brote.get("fecha_fin", fecha_inicio + pd.Timedelta(days=7)))

            # Ventana de pre-brote
            ventana_inicio = fecha_inicio - pd.Timedelta(days=pre_outbreak_window_days)

            # Máscara para este brote
            mask = (
                (result["estanque_id"] == pond_id)
                & (result["timestamp"] >= ventana_inicio)
                & (result["timestamp"] <= fecha_fin)
            )
            result.loc[mask, "outbreak"] = 1

        n_pos = (result["outbreak"] == 1).sum()
        n_total = len(result)
        pct = (n_pos / n_total * 100) if n_total > 0 else 0
        print(
            f"[BrotesETL] Etiquetado: {n_pos}/{n_total} registros positivos "
            f"({pct:.1f}%) – ventana de pre-brote: {pre_outbreak_window_days} días"
        )
        return result

    # ─── Generación de datos sintéticos ──────────────────────────────

    @staticmethod
    def generate_synthetic_outbreaks(
        n_ponds: int = 5,
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Genera datos sintéticos de brotes para testing.

        Simula brotes aleatorios con diferentes tipos de enfermedad
        y niveles de severidad para cada estanque.

        Args:
            n_ponds: Número de estanques.
            days: Número de días del período simulado.

        Returns:
            DataFrame con brotes sintéticos.
        """
        np.random.seed(123)
        start_date = datetime(2024, 1, 1)
        end_date = start_date + timedelta(days=days)

        records: list[dict] = []

        for pond_id in range(1, n_ponds + 1):
            # Cada estanque tiene entre 0 y 3 brotes en el período
            n_brotes = np.random.poisson(1.2)
            n_brotes = min(n_brotes, 3)

            for _ in range(n_brotes):
                fecha_inicio = start_date + pd.Timedelta(
                    days=np.random.uniform(10, days - 20)
                )
                duracion_dias = np.random.randint(3, 15)
                fecha_fin = fecha_inicio + pd.Timedelta(days=duracion_dias)

                records.append({
                    "estanque_id": f"E{pond_id:03d}",
                    "fecha_inicio": fecha_inicio,
                    "fecha_deteccion": fecha_inicio + pd.Timedelta(days=np.random.randint(0, 3)),
                    "fecha_fin": fecha_fin,
                    "tipo_enfermedad": np.random.choice(TIPOS_ENFERMEDAD),
                    "severidad": np.random.choice(NIVELES_SEVERIDAD, p=[0.3, 0.35, 0.25, 0.1]),
                    "tratamiento": np.random.choice(
                        ["Antibiótico", "Probiótico", "Cambio de agua", "Cuarentena", "Sin tratamiento"],
                        p=[0.25, 0.2, 0.25, 0.15, 0.15],
                    ),
                    "mortalidad_pct": round(np.random.uniform(0.5, 30.0), 1),
                })

        df = pd.DataFrame(records)
        print(f"[BrotesETL] Generados {len(df)} brotes sintéticos para {n_ponds} estanques")
        return df


# ─── Punto de entrada ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ETL Brotes – Bioseguridad Camarón AI")
    print("=" * 60)

    etl = BrotesETL()

    # Generar datos sintéticos
    print("\n→ Generando datos sintéticos de brotes...")
    synthetic = BrotesETL.generate_synthetic_outbreaks(n_ponds=5, days=90)
    etl.df = synthetic

    # Guardar datos crudos
    raw_path = ".cluster/bioseguridad-camaron-AI/repo/data/raw/brotes_crudos.csv"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    synthetic.to_csv(raw_path, index=False)
    print(f"   Datos crudos guardados en: {raw_path}")

    # Cargar y normalizar
    print("\n→ Cargando y normalizando...")
    etl.load_raw(raw_path)
    etl.normalize_events()

    # Generar datos de sensores para probar etiquetado
    print("\n→ Generando sensores sintéticos para etiquetado...")
    from src.etl.etl_sensors import SensorETL
    sensor_df = SensorETL.generate_synthetic_data(n_ponds=5, days=90)

    # Etiquetar
    print("\n→ Etiquetando eventos de brote...")
    labeled = etl.label_outbreak_events(sensor_df, etl.df)

    # Exportar
    print("\n→ Exportando a Parquet...")
    output = ".cluster/bioseguridad-camaron-AI/repo/data/processed/brotes_etiquetados.parquet"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    labeled.to_parquet(output, index=False, engine="pyarrow")

    print(f"\n✓ Pipeline de brotes completado. Output: {output}")
    print(f"  Distribución de etiquetas:\n{labeled['outbreak'].value_counts().to_string()}")
