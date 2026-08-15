#!/usr/bin/env python3
"""
Orquestador del pipeline ETL para el sistema de bioseguridad camaronera.

Ejecuta los 4 módulos ETL en secuencia:
  1. Sensores (pH, OD, salinidad, turbidez, temperatura, amoníaco)
  2. Brotes (historial de eventos de enfermedad)
  3. Operaciones (densidad, alimentación, tratamientos)
  4. Clima (temperatura ambiente, humedad, precipitación, viento)

Genera datos sintéticos, procesa cada módulo y fusiona todo en un
archivo Parquet final: data/processed/merged_features.parquet

Uso:
    python scripts/run_etl.py [--n-ponds 5] [--days 90]

Autor: Equipo de Bioseguridad Camarón AI
Fecha: 2026-08-10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Añadir el directorio src al path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.etl_sensors import SensorETL
from src.etl.etl_brotes import BrotesETL
from src.etl.etl_operations import OperationsETL
from src.etl.etl_climate import ClimateETL


# ─── Rutas base ────────────────────────────────────────────────────────

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def run_etl_pipeline(n_ponds: int, days: int) -> pd.DataFrame:
    """
    Ejecuta el pipeline ETL completo.

    Args:
        n_ponds: Número de estanques a simular.
        days: Número de días de datos.

    Returns:
        DataFrame final fusionado con todas las características.
    """
    start_time = time.time()

    # Crear directorios
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  🐐  Pipeline ETL – Bioseguridad Camarón AI")
    print(f"  Estanques: {n_ponds} | Días: {days} | Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ─── 1. ETL Sensores ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [1/4] ETL Sensores")
    print("─" * 70)

    sensor_etl = SensorETL()
    sensor_df = SensorETL.generate_synthetic_data(n_ponds=n_ponds, days=days)
    sensor_etl.df = sensor_df

    # Guardar raw
    raw_sensors = DATA_RAW / "sensores_crudos.csv"
    sensor_df.to_csv(raw_sensors, index=False)
    print(f"  → Datos crudos guardados: {raw_sensors}")

    # Procesar
    sensor_etl.detect_sensor_failures()
    sensor_etl.resample_15min()
    sensor_etl.sync_timestamps()
    sensor_etl.to_parquet(str(DATA_PROCESSED / "sensores.parquet"))

    # ─── 2. ETL Brotes ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [2/4] ETL Brotes")
    print("─" * 70)

    brotes_etl = BrotesETL()
    brotes_df = BrotesETL.generate_synthetic_outbreaks(n_ponds=n_ponds, days=days)
    brotes_etl.df = brotes_df

    # Guardar raw
    raw_brotes = DATA_RAW / "brotes_crudos.csv"
    brotes_df.to_csv(raw_brotes, index=False)
    print(f"  → Datos crudos guardados: {raw_brotes}")

    # Normalizar
    brotes_etl.normalize_events()

    # Etiquetar sensores con brotes
    sensor_labeled = brotes_etl.label_outbreak_events(
        sensor_etl.df,
        brotes_etl.df,
        pre_outbreak_window_days=7,
    )

    # Guardar datos etiquetados
    sensor_labeled.to_parquet(str(DATA_PROCESSED / "brotes_etiquetados.parquet"), index=False, engine="pyarrow")
    print(f"  → Brotes etiquetados exportados: {DATA_PROCESSED / 'brotes_etiquetados.parquet'}")

    # Actualizar sensor_df con etiquetas
    sensor_etl.df = sensor_labeled

    # ─── 3. ETL Operaciones ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [3/4] ETL Operaciones")
    print("─" * 70)

    ops_etl = OperationsETL()
    ops_df = OperationsETL.generate_synthetic_operations(n_ponds=n_ponds, days=days)
    ops_etl.df = ops_df

    # Guardar raw
    raw_ops = DATA_RAW / "operaciones_crudas.csv"
    ops_df.to_csv(raw_ops, index=False)
    print(f"  → Datos crudos guardados: {raw_ops}")

    # Normalizar
    ops_etl.normalize_operations()

    # Fusionar con sensores
    merged_ops = ops_etl.merge_with_sensors(sensor_etl.df)

    # Guardar
    merged_ops.to_parquet(str(DATA_PROCESSED / "operaciones.parquet"), index=False, engine="pyarrow")
    print(f"  → Operaciones exportadas: {DATA_PROCESSED / 'operaciones.parquet'}")

    # ─── 4. ETL Clima ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [4/4] ETL Clima")
    print("─" * 70)

    climate_etl = ClimateETL()
    climate_df = ClimateETL.generate_synthetic_climate(n_ponds=n_ponds, days=days)
    climate_etl.df = climate_df

    # Guardar raw
    raw_climate = DATA_RAW / "clima_crudo.csv"
    climate_df.to_csv(raw_climate, index=False)
    print(f"  → Datos crudos guardados: {raw_climate}")

    # Resamplear
    climate_etl.resample_15min()

    # Fusionar con el dataframe actual (que ya tiene sensores + operaciones + brotes)
    final_df = climate_etl.merge_with_sensors(merged_ops)

    # ─── Guardar resultado final ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Fusión final")
    print("=" * 70)

    output_path = DATA_PROCESSED / "merged_features.parquet"
    final_df.to_parquet(str(output_path), index=False, engine="pyarrow")

    elapsed = time.time() - start_time
    print(f"\n✅ Pipeline completado en {elapsed:.1f}s")
    print(f"   Archivo final: {output_path}")
    print(f"   Registros: {len(final_df):,}")
    print(f"   Columnas: {len(final_df.columns)}")
    print(f"   Estanques: {final_df['estanque_id'].nunique()}")
    print(f"   Rango temporal: {final_df['timestamp'].min()} → {final_df['timestamp'].max()}")

    if "outbreak" in final_df.columns:
        pos = (final_df["outbreak"] == 1).sum()
        total = len(final_df)
        print(f"   Etiquetas outbreak: {pos:,} positivas / {total:,} total ({pos/total*100:.1f}%)")

    print(f"\n   Columnas del dataset final:")
    for i, col in enumerate(final_df.columns, 1):
        print(f"     {i:2d}. {col}")

    return final_df


def main() -> None:
    """Punto de entrada con argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline ETL para sistema de bioseguridad camaronera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/run_etl.py
  python scripts/run_etl.py --n-ponds 10 --days 180
  python scripts/run_etl.py --n-ponds 3 --days 30
        """,
    )
    parser.add_argument(
        "--n-ponds",
        type=int,
        default=5,
        help="Número de estanques a simular (default: 5)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Número de días de datos (default: 90)",
    )
    args = parser.parse_args()

    if args.n_ponds < 1:
        parser.error("--n-ponds debe ser >= 1")
    if args.days < 1:
        parser.error("--days debe ser >= 1")

    run_etl_pipeline(n_ponds=args.n_ponds, days=args.days)


if __name__ == "__main__":
    main()
