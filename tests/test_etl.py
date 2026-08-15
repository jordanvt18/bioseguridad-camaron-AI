"""
Tests para los módulos ETL del sistema de bioseguridad camaronera.

Ejecuta con:
    pytest tests/test_etl.py -v

Autor: Equipo de Bioseguridad Camarón AI
Fecha: 2026-08-10
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Añadir src al path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.etl_sensors import SensorETL, SENSOR_RANGES
from src.etl.etl_brotes import BrotesETL
from src.etl.etl_operations import OperationsETL
from src.etl.etl_climate import ClimateETL


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def small_sensor_df() -> pd.DataFrame:
    """Genera un DataFrame pequeño de sensores para tests rápidos."""
    return SensorETL.generate_synthetic_data(n_ponds=2, days=7)


@pytest.fixture
def small_outbreak_df() -> pd.DataFrame:
    """Genera un DataFrame pequeño de brotes."""
    return BrotesETL.generate_synthetic_outbreaks(n_ponds=2, days=7)


@pytest.fixture
def small_operations_df() -> pd.DataFrame:
    """Genera un DataFrame pequeño de operaciones."""
    return OperationsETL.generate_synthetic_operations(n_ponds=2, days=7)


@pytest.fixture
def small_climate_df() -> pd.DataFrame:
    """Genera un DataFrame pequeño de clima."""
    return ClimateETL.generate_synthetic_climate(n_ponds=2, days=7)


# ═══════════════════════════════════════════════════════════════════════
#  Tests: SensorETL
# ═══════════════════════════════════════════════════════════════════════


class TestSensorETL:
    """Tests para el módulo ETL de sensores."""

    def test_generate_synthetic_data_columns(self, small_sensor_df: pd.DataFrame) -> None:
        """Verifica que los datos sintéticos tengan todas las columnas esperadas."""
        expected_cols = {"estanque_id", "timestamp", "ph", "oxigeno_disuelto",
                         "temperatura", "salinidad", "turbidez", "amonio"}
        assert expected_cols.issubset(set(small_sensor_df.columns))

    def test_generate_synthetic_data_shape(self, small_sensor_df: pd.DataFrame) -> None:
        """Verifica el tamaño del DataFrame sintético."""
        # 2 estanques × 7 días × 24h × 4 lecturas/h = 1344
        assert len(small_sensor_df) == 2 * 7 * 24 * 4

    def test_generate_synthetic_data_ranges(self, small_sensor_df: pd.DataFrame) -> None:
        """Verifica que la mayoría de los valores estén en rangos físicos."""
        for col, (min_val, max_val) in SENSOR_RANGES.items():
            valid = small_sensor_df[col].dropna()
            # Al menos el 95% deben estar en rango (permitimos fallos inyectados)
            in_range = ((valid >= min_val) & (valid <= max_val)).mean()
            assert in_range > 0.95, f"{col}: solo {in_range:.1%} en rango [{min_val}, {max_val}]"

    def test_load_raw_csv(self, small_sensor_df: pd.DataFrame, tmp_path: Path) -> None:
        """Verifica la carga desde CSV."""
        path = tmp_path / "sensors.csv"
        small_sensor_df.to_csv(path, index=False)

        etl = SensorETL()
        loaded = etl.load_raw(str(path))
        assert len(loaded) == len(small_sensor_df)
        assert "timestamp" in loaded.columns
        assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])

    def test_load_raw_parquet(self, small_sensor_df: pd.DataFrame, tmp_path: Path) -> None:
        """Verifica la carga desde Parquet."""
        path = tmp_path / "sensors.parquet"
        small_sensor_df.to_parquet(path, index=False)

        etl = SensorETL()
        loaded = etl.load_raw(str(path))
        assert len(loaded) == len(small_sensor_df)

    def test_load_raw_not_found(self) -> None:
        """Verifica error cuando el archivo no existe."""
        etl = SensorETL()
        with pytest.raises(FileNotFoundError):
            etl.load_raw("/no/existe/archivo.csv")

    def test_load_raw_bad_extension(self, tmp_path: Path) -> None:
        """Verifica error con extensión no soportada."""
        path = tmp_path / "sensors.txt"
        path.write_text("dummy")
        etl = SensorETL()
        with pytest.raises(ValueError, match="Extensión no soportada"):
            etl.load_raw(str(path))

    def test_resample_15min(self, small_sensor_df: pd.DataFrame) -> None:
        """Verifica que el resampleo produce intervalos de 15 min."""
        etl = SensorETL()
        etl.df = small_sensor_df
        resampled = etl.resample_15min()

        for pond_id, group in resampled.groupby("estanque_id"):
            group = group.sort_values("timestamp")
            diffs = group["timestamp"].diff().dropna()
            # Todas las diferencias deben ser de 15 minutos
            assert (diffs == pd.Timedelta(minutes=15)).all(), \
                f"Intervalos irregulares en estanque {pond_id}"

    def test_detect_sensor_failures_out_of_range(self) -> None:
        """Verifica detección de valores fuera de rango."""
        df = pd.DataFrame({
            "estanque_id": ["E001"] * 5,
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="15min"),
            "ph": [7.5, 7.6, -1.0, 7.4, 15.0],       # -1.0 y 15.0 fuera de rango
            "oxigeno_disuelto": [6.0, 6.1, 6.2, 6.0, 6.1],
            "temperatura": [28.0, 28.1, 28.2, 28.0, 28.1],
            "salinidad": [25.0, 25.1, 25.0, 25.1, 25.0],
            "turbidez": [50.0, 51.0, 50.0, 51.0, 50.0],
            "amonio": [0.5, 0.5, 0.5, 0.5, 0.5],
        })
        etl = SensorETL()
        etl.df = df
        failures = etl.detect_sensor_failures()

        out_of_range = failures[failures["fallo_tipo"] == "fuera_de_rango"]
        assert len(out_of_range) == 2
        assert "ph" in out_of_range["variable"].values

    def test_detect_sensor_failures_stuck(self) -> None:
        """Verifica detección de sensor atascado."""
        # 5 lecturas idénticas de pH (stuck)
        df = pd.DataFrame({
            "estanque_id": ["E001"] * 6,
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="15min"),
            "ph": [7.5, 7.5, 7.5, 7.5, 7.5, 7.6],
            "oxigeno_disuelto": [6.0, 6.1, 6.2, 6.3, 6.4, 6.5],
            "temperatura": [28.0, 28.1, 28.2, 28.3, 28.4, 28.5],
            "salinidad": [25.0, 25.1, 25.0, 25.1, 25.0, 25.1],
            "turbidez": [50.0, 51.0, 50.0, 51.0, 50.0, 51.0],
            "amonio": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        })
        etl = SensorETL()
        etl.df = df
        failures = etl.detect_sensor_failures()

        stuck = failures[failures["fallo_tipo"] == "sensor_atascado"]
        assert len(stuck) > 0
        assert "ph" in stuck["variable"].values

    def test_detect_sensor_failures_time_gap(self) -> None:
        """Verifica detección de huecos temporales > 1h."""
        timestamps = pd.date_range("2024-01-01", periods=5, freq="15min")
        # Insertar un hueco de 2 horas
        timestamps = timestamps.insert(3, timestamps[2] + pd.Timedelta(hours=2))

        df = pd.DataFrame({
            "estanque_id": ["E001"] * 6,
            "timestamp": timestamps,
            "ph": [7.5] * 6,
            "oxigeno_disuelto": [6.0] * 6,
            "temperatura": [28.0] * 6,
            "salinidad": [25.0] * 6,
            "turbidez": [50.0] * 6,
            "amonio": [0.5] * 6,
        })
        etl = SensorETL()
        etl.df = df
        failures = etl.detect_sensor_failures()

        gaps = failures[failures["fallo_tipo"] == "hueco_temporal"]
        assert len(gaps) >= 1
        assert (gaps["valor"] >= 60).all()  # > 60 minutos

    def test_sync_timestamps(self, small_sensor_df: pd.DataFrame) -> None:
        """Verifica que la sincronización alinea timestamps entre estanques."""
        etl = SensorETL()
        etl.df = small_sensor_df.copy()
        synced = etl.sync_timestamps()

        # Todos los estanques deben tener los mismos timestamps
        for pond_id in synced["estanque_id"].unique():
            ts = synced[synced["estanque_id"] == pond_id]["timestamp"]
            assert ts.is_monotonic_increasing

        # Verificar que ambos estanques comparten timestamps
        ts1 = set(synced[synced["estanque_id"] == "E001"]["timestamp"])
        ts2 = set(synced[synced["estanque_id"] == "E002"]["timestamp"])
        assert ts1 == ts2, "Los timestamps no están sincronizados entre estanques"

    def test_to_parquet(self, small_sensor_df: pd.DataFrame, tmp_path: Path) -> None:
        """Verifica exportación a Parquet."""
        etl = SensorETL()
        etl.df = small_sensor_df
        output = tmp_path / "output.parquet"
        etl.to_parquet(str(output))

        assert output.exists()
        loaded = pd.read_parquet(output)
        assert len(loaded) == len(small_sensor_df)


# ═══════════════════════════════════════════════════════════════════════
#  Tests: BrotesETL
# ═══════════════════════════════════════════════════════════════════════


class TestBrotesETL:
    """Tests para el módulo ETL de brotes."""

    def test_generate_synthetic_outbreaks_columns(self, small_outbreak_df: pd.DataFrame) -> None:
        """Verifica columnas del DataFrame de brotes sintéticos."""
        expected = {"estanque_id", "fecha_inicio", "tipo_enfermedad", "severidad"}
        assert expected.issubset(set(small_outbreak_df.columns))

    def test_generate_synthetic_outbreaks_not_empty(self) -> None:
        """Verifica que con suficientes estanques se generan brotes."""
        df = BrotesETL.generate_synthetic_outbreaks(n_ponds=10, days=90)
        assert len(df) > 0, "Debería generar al menos un brote con 10 estanques"

    def test_load_raw(self, small_outbreak_df: pd.DataFrame, tmp_path: Path) -> None:
        """Verifica carga de datos de brotes."""
        path = tmp_path / "brotes.csv"
        small_outbreak_df.to_csv(path, index=False)

        etl = BrotesETL()
        loaded = etl.load_raw(str(path))
        assert len(loaded) == len(small_outbreak_df)

    def test_normalize_events(self, small_outbreak_df: pd.DataFrame) -> None:
        """Verifica normalización de eventos de brote."""
        etl = BrotesETL()
        etl.df = small_outbreak_df.copy()
        normalized = etl.normalize_events()

        assert "severidad_num" in normalized.columns
        assert normalized["severidad_num"].between(0, 3).all()
        assert normalized["tipo_enfermedad"].str.istitle().all()

    def test_label_outbreak_events(self) -> None:
        """Verifica el etiquetado de brotes en datos de sensor."""
        # Crear datos de sensor simples
        sensor_df = pd.DataFrame({
            "estanque_id": ["E001"] * 100,
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="15min"),
            "ph": [7.5] * 100,
            "oxigeno_disuelto": [6.0] * 100,
            "temperatura": [28.0] * 100,
            "salinidad": [25.0] * 100,
            "turbidez": [50.0] * 100,
            "amonio": [0.5] * 100,
        })

        # Crear un brote que empieza en el medio de los datos
        outbreak_df = pd.DataFrame({
            "estanque_id": ["E001"],
            "fecha_inicio": [pd.Timestamp("2024-01-01 12:00")],
            "fecha_fin": [pd.Timestamp("2024-01-01 18:00")],
            "tipo_enfermedad": ["Mancha Blanca"],
            "severidad": ["severa"],
            "tratamiento": ["Cuarentena"],
            "mortalidad_pct": [15.0],
        })

        etl = BrotesETL()
        labeled = etl.label_outbreak_events(
            sensor_df, outbreak_df, pre_outbreak_window_days=7
        )

        assert "outbreak" in labeled.columns
        # Debe haber al menos algunos positivos (ventana de 7 días + duración del brote)
        assert (labeled["outbreak"] == 1).sum() > 0
        # La mayoría serán 0 (100 registros = ~25h, ventana 7 días debería capturar varios)
        assert (labeled["outbreak"] == 0).sum() > 0

    def test_label_outbreak_events_no_brotes(self) -> None:
        """Verifica etiquetado cuando no hay brotes."""
        sensor_df = pd.DataFrame({
            "estanque_id": ["E001"] * 10,
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="15min"),
            "ph": [7.5] * 10,
        })
        outbreak_df = pd.DataFrame(columns=["estanque_id", "fecha_inicio", "fecha_fin"])

        etl = BrotesETL()
        labeled = etl.label_outbreak_events(sensor_df, outbreak_df)
        assert (labeled["outbreak"] == 0).all()


# ═══════════════════════════════════════════════════════════════════════
#  Tests: OperationsETL
# ═══════════════════════════════════════════════════════════════════════


class TestOperationsETL:
    """Tests para el módulo ETL de operaciones."""

    def test_generate_synthetic_operations_columns(self, small_operations_df: pd.DataFrame) -> None:
        """Verifica columnas del DataFrame de operaciones."""
        expected = {"estanque_id", "fecha", "densidad_siembra",
                    "tasa_alimentacion", "peso_promedio", "edad_dias"}
        assert expected.issubset(set(small_operations_df.columns))

    def test_generate_synthetic_operations_shape(self, small_operations_df: pd.DataFrame) -> None:
        """Verifica el tamaño del DataFrame."""
        # 2 estanques × 7 días = 14 registros
        assert len(small_operations_df) == 2 * 7

    def test_load_raw(self, small_operations_df: pd.DataFrame, tmp_path: Path) -> None:
        """Verifica carga de datos operacionales."""
        path = tmp_path / "ops.csv"
        small_operations_df.to_csv(path, index=False)

        etl = OperationsETL()
        loaded = etl.load_raw(str(path))
        assert len(loaded) == len(small_operations_df)

    def test_normalize_operations(self, small_operations_df: pd.DataFrame) -> None:
        """Verifica normalización de operaciones."""
        etl = OperationsETL()
        etl.df = small_operations_df.copy()
        normalized = etl.normalize_operations()

        assert normalized["densidad_siembra"].notna().all()
        assert normalized["tasa_alimentacion"].notna().all()
        assert (normalized["densidad_siembra"] >= 0).all()
        assert (normalized["tasa_alimentacion"] >= 0).all()

    def test_merge_with_sensors(self, small_sensor_df: pd.DataFrame,
                                small_operations_df: pd.DataFrame) -> None:
        """Verifica la fusión de operaciones con sensores."""
        etl = OperationsETL()
        etl.df = small_operations_df.copy()
        etl.normalize_operations()

        merged = etl.merge_with_sensors(small_sensor_df)

        # Debe tener todas las columnas de sensores + operaciones
        sensor_cols = set(small_sensor_df.columns)
        ops_cols = {"densidad_siembra", "tasa_alimentacion", "peso_promedio", "edad_dias"}
        assert sensor_cols.issubset(set(merged.columns))
        assert ops_cols.issubset(set(merged.columns))
        # El número de filas no debe disminuir
        assert len(merged) >= len(small_sensor_df)


# ═══════════════════════════════════════════════════════════════════════
#  Tests: ClimateETL
# ═══════════════════════════════════════════════════════════════════════


class TestClimateETL:
    """Tests para el módulo ETL de clima."""

    def test_generate_synthetic_climate_columns(self, small_climate_df: pd.DataFrame) -> None:
        """Verifica columnas del DataFrame climático."""
        expected = {"estanque_id", "timestamp", "temp_ambiente",
                    "humedad_relativa", "precipitacion", "velocidad_viento"}
        assert expected.issubset(set(small_climate_df.columns))

    def test_generate_synthetic_climate_shape(self, small_climate_df: pd.DataFrame) -> None:
        """Verifica el tamaño del DataFrame."""
        # 2 estanques × 7 días × 24h = 336 registros (datos horarios)
        assert len(small_climate_df) == 2 * 7 * 24

    def test_generate_synthetic_climate_ranges(self, small_climate_df: pd.DataFrame) -> None:
        """Verifica rangos de variables climáticas."""
        assert (small_climate_df["humedad_relativa"] >= 0).all()
        assert (small_climate_df["humedad_relativa"] <= 100).all()
        assert (small_climate_df["precipitacion"] >= 0).all()
        assert (small_climate_df["velocidad_viento"] >= 0).all()

    def test_load_raw(self, small_climate_df: pd.DataFrame, tmp_path: Path) -> None:
        """Verifica carga de datos climáticos."""
        path = tmp_path / "climate.csv"
        small_climate_df.to_csv(path, index=False)

        etl = ClimateETL()
        loaded = etl.load_raw(str(path))
        assert len(loaded) == len(small_climate_df)

    def test_resample_15min(self, small_climate_df: pd.DataFrame) -> None:
        """Verifica resampleo a 15 minutos."""
        etl = ClimateETL()
        etl.df = small_climate_df.copy()
        resampled = etl.resample_15min()

        for pond_id, group in resampled.groupby("estanque_id"):
            group = group.sort_values("timestamp")
            diffs = group["timestamp"].diff().dropna()
            assert (diffs == pd.Timedelta(minutes=15)).all()

    def test_merge_with_sensors(self, small_sensor_df: pd.DataFrame,
                                small_climate_df: pd.DataFrame) -> None:
        """Verifica la fusión de datos climáticos con sensores."""
        etl = ClimateETL()
        etl.df = small_climate_df.copy()
        etl.resample_15min()

        merged = etl.merge_with_sensors(small_sensor_df)

        climate_cols = {"temp_ambiente", "humedad_relativa", "precipitacion", "velocidad_viento"}
        assert climate_cols.issubset(set(merged.columns))
        assert len(merged) >= len(small_sensor_df)

    def test_merge_with_sensors_no_nans(self, small_sensor_df: pd.DataFrame,
                                        small_climate_df: pd.DataFrame) -> None:
        """Verifica que no haya NaN en columnas climáticas tras la fusión."""
        etl = ClimateETL()
        etl.df = small_climate_df.copy()
        etl.resample_15min()

        merged = etl.merge_with_sensors(small_sensor_df)

        climate_cols = ["temp_ambiente", "humedad_relativa", "precipitacion", "velocidad_viento"]
        for col in climate_cols:
            if col in merged.columns:
                # Permitir algunos NaN si los timestamps no se solapan
                na_pct = merged[col].isna().mean()
                assert na_pct < 0.1, f"{col} tiene {na_pct:.1%} NaN tras fusión"


# ═══════════════════════════════════════════════════════════════════════
#  Tests: Integración del pipeline completo
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
    """Tests de integración del pipeline ETL completo."""

    def test_full_pipeline_merge(self) -> None:
        """Verifica que todos los módulos se integran correctamente."""
        n_ponds, days = 2, 7

        # 1. Sensores
        sensor_etl = SensorETL()
        sensor_df = SensorETL.generate_synthetic_data(n_ponds=n_ponds, days=days)
        sensor_etl.df = sensor_df
        sensor_etl.resample_15min()

        # 2. Brotes
        brotes_etl = BrotesETL()
        brotes_df = BrotesETL.generate_synthetic_outbreaks(n_ponds=n_ponds, days=days)
        brotes_etl.df = brotes_df
        brotes_etl.normalize_events()
        labeled = brotes_etl.label_outbreak_events(sensor_etl.df, brotes_etl.df)

        assert "outbreak" in labeled.columns
        sensor_etl.df = labeled

        # 3. Operaciones
        ops_etl = OperationsETL()
        ops_df = OperationsETL.generate_synthetic_operations(n_ponds=n_ponds, days=days)
        ops_etl.df = ops_df
        ops_etl.normalize_operations()
        merged_ops = ops_etl.merge_with_sensors(sensor_etl.df)

        assert "densidad_siembra" in merged_ops.columns
        assert "outbreak" in merged_ops.columns

        # 4. Clima
        climate_etl = ClimateETL()
        climate_df = ClimateETL.generate_synthetic_climate(n_ponds=n_ponds, days=days)
        climate_etl.df = climate_df
        climate_etl.resample_15min()
        final_df = climate_etl.merge_with_sensors(merged_ops)

        # Verificar que el DataFrame final tiene todas las columnas
        expected_sensor_cols = {"ph", "oxigeno_disuelto", "temperatura", "salinidad", "turbidez", "amonio"}
        expected_ops_cols = {"densidad_siembra", "tasa_alimentacion"}
        expected_climate_cols = {"temp_ambiente", "humedad_relativa", "precipitacion", "velocidad_viento"}
        expected_label = {"outbreak"}

        all_expected = expected_sensor_cols | expected_ops_cols | expected_climate_cols | expected_label
        assert all_expected.issubset(set(final_df.columns))
        assert len(final_df) > 0

    def test_pipeline_output_to_parquet(self, tmp_path: Path) -> None:
        """Verifica que el resultado final se puede escribir a Parquet."""
        n_ponds, days = 2, 3

        sensor_df = SensorETL.generate_synthetic_data(n_ponds=n_ponds, days=days)
        sensor_etl = SensorETL()
        sensor_etl.df = sensor_df
        sensor_etl.resample_15min()

        brotes_etl = BrotesETL()
        brotes_df = BrotesETL.generate_synthetic_outbreaks(n_ponds=n_ponds, days=days)
        brotes_etl.df = brotes_df
        brotes_etl.normalize_events()
        labeled = brotes_etl.label_outbreak_events(sensor_etl.df, brotes_etl.df)

        ops_etl = OperationsETL()
        ops_df = OperationsETL.generate_synthetic_operations(n_ponds=n_ponds, days=days)
        ops_etl.df = ops_df
        ops_etl.normalize_operations()
        merged = ops_etl.merge_with_sensors(labeled)

        climate_etl = ClimateETL()
        climate_df = ClimateETL.generate_synthetic_climate(n_ponds=n_ponds, days=days)
        climate_etl.df = climate_df
        climate_etl.resample_15min()
        final = climate_etl.merge_with_sensors(merged)

        output = tmp_path / "merged_features.parquet"
        final.to_parquet(output, index=False, engine="pyarrow")

        assert output.exists()
        loaded = pd.read_parquet(output)
        assert len(loaded) == len(final)
