"""
Script de entrenamiento para el sistema de detección de brotes.

Flujo:
    1. Carga datos procesados (data/processed/merged_features.parquet)
       o genera datos sintéticos si no existe.
    2. Crea secuencias con ventana deslizante (lookback=96, horizon=1).
    3. Divide train/val/test (70/15/15) cronológicamente.
    4. Entrena detector (LSTM o Transformer) y Autoencoder.
    5. Evalúa en test set.
    6. Guarda modelos en models/.

Uso:
    python -m src.models.train --model lstm --epochs 50
    python -m src.models.train --model transformer --epochs 80
    python -m src.models.train --model lstm --synthetic  # forzar datos sintéticos
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Asegurar que el directorio src esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.lstm_detector import LSTMOutbreakDetector
from src.models.transformer_detector import TransformerOutbreakDetector
from src.models.autoencoder_anomaly import SensorAutoencoder
from src.models.outbreak_classifier import OutbreakClassifier


# ===========================================================================
# Configuración
# ===========================================================================
LOOKBACK = 96  # 24 horas a intervalos de 15 minutos
INPUT_DIM = 6  # 6 sensores: temperatura, pH, oxígeno, salinidad, turbidez, amonio
DATA_PATH = "data/processed/merged_features.parquet"
MODELS_DIR = "models"
SENSOR_COLUMNS = ["temperatura", "ph", "oxigeno", "salinidad", "turbidez", "amonio"]
LABEL_COLUMN = "brote"


# ===========================================================================
# Datos sintéticos
# ===========================================================================
def generate_synthetic_data(
    n_days: int = 90,
    interval_min: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera datos sintéticos de sensores para demostración.

    Simula 6 sensores con patrones diarios, ruido y brotes
    inyectados aleatoriamente.

    Args:
        n_days: Número de días a simular.
        interval_min: Intervalo entre mediciones (minutos).
        seed: Semilla aleatoria.

    Returns:
        DataFrame con columnas de sensores y etiqueta 'brote'.
    """
    rng = np.random.default_rng(seed)
    n_samples = n_days * 24 * (60 // interval_min)
    time_index = pd.date_range(
        start="2024-01-01", periods=n_samples, freq=f"{interval_min}min"
    )

    # Patrones base con ciclo diario
    t = np.arange(n_samples)
    day_cycle = np.sin(2 * np.pi * t / (24 * 4))  # ciclo de 24h

    # Sensores con patrones realistas
    temperatura = 28 + 2 * day_cycle + rng.normal(0, 0.3, n_samples)
    ph = 7.8 + 0.3 * day_cycle + rng.normal(0, 0.1, n_samples)
    oxigeno = 5.5 + 1.0 * day_cycle + rng.normal(0, 0.2, n_samples)
    salinidad = 25 + 2 * rng.normal(0, 1, n_samples) * 0.1
    turbidez = 15 + 5 * np.abs(day_cycle) + rng.normal(0, 1, n_samples)
    amonio = 0.1 + 0.05 * rng.normal(0, 1, n_samples)

    # Inyectar brotes: períodos donde los sensores se alteran
    brote = np.zeros(n_samples, dtype=int)
    n_outbreaks = max(3, n_days // 20)
    outbreak_duration = 48  # 12 horas

    for _ in range(n_outbreaks):
        start = rng.integers(LOOKBACK, n_samples - outbreak_duration)
        brote[start : start + outbreak_duration] = 1

        # Alterar sensores durante el brote
        temperatura[start : start + outbreak_duration] += rng.uniform(1.5, 3.5)
        ph[start : start + outbreak_duration] -= rng.uniform(0.5, 1.5)
        oxigeno[start : start + outbreak_duration] -= rng.uniform(1.5, 3.0)
        turbidez[start : start + outbreak_duration] += rng.uniform(10, 25)
        amonio[start : start + outbreak_duration] += rng.uniform(0.3, 0.8)

    df = pd.DataFrame(
        {
            "timestamp": time_index,
            "temperatura": temperatura,
            "ph": ph,
            "oxigeno": oxigeno,
            "salinidad": salinidad,
            "turbidez": turbidez,
            "amonio": amonio,
            "brote": brote,
        }
    )
    print(f"Datos sintéticos generados: {len(df)} muestras, {brote.sum()} con brote.")
    return df


# ===========================================================================
# Utilidades
# ===========================================================================
def create_sequences(
    data: np.ndarray, labels: np.ndarray, lookback: int = LOOKBACK
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crea secuencias con ventana deslizante.

    Args:
        data: Array de forma (n_samples, input_dim).
        labels: Array de forma (n_samples,).
        lookback: Tamaño de la ventana.

    Returns:
        Tupla (X_seq, y_seq) donde X_seq tiene forma (n_seq, lookback, input_dim)
        y y_seq tiene forma (n_seq,).
    """
    X_seq, y_seq = [], []
    for i in range(len(data) - lookback + 1):
        X_seq.append(data[i : i + lookback])
        y_seq.append(labels[i + lookback - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def normalize_data(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normaliza los datos usando estadísticas del set de entrenamiento.

    Args:
        train, val, test: Arrays a normalizar.

    Returns:
        Tupla de arrays normalizados.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    return (train - mean) / std, (val - mean) / std, (test - mean) / std


# ===========================================================================
# Entrenamiento principal
# ===========================================================================
def train_detector(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    input_dim: int,
) -> LSTMOutbreakDetector | TransformerOutbreakDetector:
    """Entrena el detector temporal seleccionado."""
    if model_type == "lstm":
        model = LSTMOutbreakDetector(input_dim=input_dim)
    else:
        model = TransformerOutbreakDetector(input_dim=input_dim)

    model.train_model(train_loader, val_loader, epochs=epochs, lr=lr)
    return model


def train_autoencoder(
    X_train_normal: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    input_dim: int,
) -> SensorAutoencoder:
    """Entrena el autoencoder con datos normales."""
    ae = SensorAutoencoder(input_dim=input_dim)
    ds = TensorDataset(torch.from_numpy(X_train_normal).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    ae.train_model(loader, epochs=epochs, lr=lr)
    ae.set_threshold(X_train_normal, percentile=95)
    return ae


def main(args: argparse.Namespace) -> None:
    """Función principal de entrenamiento."""
    print("=" * 60)
    print("  Sistema de Detección de Brotes - Camaroneras")
    print("=" * 60)

    # --- 1. Cargar datos ---
    use_synthetic = args.synthetic
    data_path = Path(args.data_path) if args.data_path else Path(DATA_PATH)

    if not use_synthetic and data_path.exists():
        print(f"\nCargando datos desde: {data_path}")
        df = pd.read_parquet(data_path)
        # Verificar columnas
        missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
        if missing:
            print(f"  Faltan columnas: {missing}. Generando datos sintéticos.")
            use_synthetic = True
        else:
            print(f"  {len(df)} muestras cargadas.")
    else:
        if not use_synthetic:
            print(f"  No se encontró {data_path}. Generando datos sintéticos.")
        use_synthetic = True

    if use_synthetic:
        df = generate_synthetic_data(n_days=args.synthetic_days)

    # --- 2. Preparar datos ---
    sensor_data = df[SENSOR_COLUMNS].values.astype(np.float32)
    labels = df[LABEL_COLUMN].values.astype(np.float32) if LABEL_COLUMN in df.columns else np.zeros(len(df), dtype=np.float32)

    # División cronológica 70/15/15
    n = len(sensor_data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train_raw = sensor_data[:train_end]
    X_val_raw = sensor_data[train_end:val_end]
    X_test_raw = sensor_data[val_end:]

    y_train_raw = labels[:train_end]
    y_val_raw = labels[train_end:val_end]
    y_test_raw = labels[val_end:]

    # Normalizar
    X_train_norm, X_val_norm, X_test_norm = normalize_data(X_train_raw, X_val_raw, X_test_raw)

    # Crear secuencias
    X_train, y_train = create_sequences(X_train_norm, y_train_raw, LOOKBACK)
    X_val, y_val = create_sequences(X_val_norm, y_val_raw, LOOKBACK)
    X_test, y_test = create_sequences(X_test_norm, y_test_raw, LOOKBACK)

    print(f"\nSecuencias creadas (lookback={LOOKBACK}):")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Brotes - Train: {y_train.sum():.0f}, Val: {y_val.sum():.0f}, Test: {y_test.sum():.0f}")

    # --- 3. DataLoaders ---
    batch_size = args.batch_size
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # --- 4. Entrenar detector temporal ---
    print(f"\n{'=' * 60}")
    print(f"  Entrenando detector: {args.model.upper()}")
    print(f"{'=' * 60}")
    detector = train_detector(
        args.model, train_loader, val_loader, args.epochs, args.lr, INPUT_DIM
    )

    # Evaluar detector en test
    test_proba = detector.predict(X_test)
    from sklearn.metrics import f1_score, roc_auc_score

    try:
        test_auc = roc_auc_score(y_test, test_proba)
    except ValueError:
        test_auc = 0.0
    test_f1 = f1_score(y_test, (test_proba >= 0.5).astype(int), zero_division=0)
    print(f"\n  Test AUC: {test_auc:.4f} | Test F1: {test_f1:.4f}")

    # --- 5. Entrenar Autoencoder ---
    print(f"\n{'=' * 60}")
    print("  Entrenando Autoencoder")
    print(f"{'=' * 60}")
    normal_mask = y_train_raw == 0
    X_train_normal = X_train_norm[normal_mask]
    autoencoder = train_autoencoder(
        X_train_normal, epochs=min(100, args.epochs * 2), lr=args.lr,
        batch_size=batch_size, input_dim=INPUT_DIM,
    )

    # Evaluar autoencoder
    test_errors = autoencoder.compute_reconstruction_error(X_test[:, -1, :])
    test_anomalies = autoencoder.detect_anomalies(X_test[:, -1, :])
    print(f"  Anomalías detectadas en test: {test_anomalies.sum()} / {len(test_anomalies)}")

    # --- 6. Clasificador integrado ---
    print(f"\n{'=' * 60}")
    print("  Evaluando clasificador integrado")
    print(f"{'=' * 60}")
    # Reconstruir datos completos normalizados para el clasificador
    X_full_norm = np.concatenate([X_train_norm, X_val_norm, X_test_norm])
    y_full = np.concatenate([y_train_raw, y_val_raw, y_test_raw])

    classifier = OutbreakClassifier(
        model_type=args.model, input_dim=INPUT_DIM, lookback=LOOKBACK,
        use_autoencoder=True, ensemble=True,
    )
    # Asignar modelos ya entrenados
    classifier.detector = detector
    classifier.autoencoder = autoencoder

    # Calcular estadísticas de normalización del autoencoder
    normal_errors = autoencoder.compute_reconstruction_error(X_train_normal)
    classifier._anomaly_mean = float(np.mean(normal_errors))
    classifier._anomaly_std = float(np.std(normal_errors) + 1e-8)

    # Evaluar en test
    X_test_full = X_test_norm
    y_test_full = y_test_raw
    metrics = classifier.evaluate(X_test_full, y_test_full)
    print(f"\n  Métricas finales en test:")
    print(f"    AUC:            {metrics['auc']:.4f}")
    print(f"    F1:             {metrics['f1']:.4f}")
    print(f"    Precision@10%:  {metrics['precision_at_k']:.4f}")

    # --- 7. Guardar modelos ---
    models_dir = Path(args.models_dir) if args.models_dir else Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    detector_path = models_dir / f"detector_{args.model}.pt"
    detector.save_model(str(detector_path))
    print(f"\n  Detector guardado en: {detector_path}")

    ae_path = models_dir / "autoencoder.pt"
    autoencoder.save_model(str(ae_path))
    print(f"  Autoencoder guardado en: {ae_path}")

    # Guardar clasificador completo
    classifier_dir = models_dir / f"classifier_{args.model}"
    classifier.save(str(classifier_dir))
    print(f"  Clasificador guardado en: {classifier_dir}")

    print(f"\n{'=' * 60}")
    print("  ¡Entrenamiento completado!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento del sistema de detección de brotes en camaroneras."
    )
    parser.add_argument(
        "--model", type=str, default="lstm", choices=["lstm", "transformer"],
        help="Tipo de detector temporal (default: lstm).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Número de épocas de entrenamiento (default: 50).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Tasa de aprendizaje (default: 1e-3).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Tamaño de batch (default: 32).",
    )
    parser.add_argument(
        "--data-path", type=str, default=DATA_PATH,
        help=f"Ruta al archivo parquet de datos (default: {DATA_PATH}).",
    )
    parser.add_argument(
        "--models-dir", type=str, default=MODELS_DIR,
        help=f"Directorio para guardar modelos (default: {MODELS_DIR}).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Forzar uso de datos sintéticos.",
    )
    parser.add_argument(
        "--synthetic-days", type=int, default=90,
        help="Número de días de datos sintéticos (default: 90).",
    )

    args = parser.parse_args()
    main(args)
