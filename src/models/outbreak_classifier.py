"""
Clasificador de brotes que combina detector temporal (LSTM/Transformer)
con detector de anomalías (Autoencoder).

Pipeline:
    1. Detector temporal → probabilidad de brote [0, 1]
    2. Autoencoder → score de anomalía (error de reconstrucción)
    3. Fusión: promedio entre probabilidad temporal y (1 - anomalía normalizada)

El resultado es una probabilidad final de brote en [0, 1].
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .lstm_detector import LSTMOutbreakDetector
from .transformer_detector import TransformerOutbreakDetector
from .autoencoder_anomaly import SensorAutoencoder


class OutbreakClassifier:
    """Clasificador de brotes que orquesta el pipeline completo."""

    def __init__(
        self,
        model_type: Literal["lstm", "transformer"] = "lstm",
        input_dim: int = 6,
        lookback: int = 96,
        use_autoencoder: bool = True,
        ensemble: bool = True,
    ) -> None:
        """
        Inicializa el clasificador de brotes.

        Args:
            model_type: Tipo de detector temporal ('lstm' o 'transformer').
            input_dim: Número de características de entrada.
            lookback: Ventana de observación (pasos temporales).
            use_autoencoder: Si True, integra el autoencoder para detección de anomalías.
            ensemble: Si True, combina la probabilidad temporal con el score de anomalía.
        """
        self.model_type = model_type
        self.input_dim = input_dim
        self.lookback = lookback
        self.use_autoencoder = use_autoencoder
        self.ensemble = ensemble

        # Detector temporal
        if model_type == "lstm":
            self.detector = LSTMOutbreakDetector(input_dim=input_dim)
        elif model_type == "transformer":
            self.detector = TransformerOutbreakDetector(input_dim=input_dim)
        else:
            raise ValueError(f"model_type debe ser 'lstm' o 'transformer', no '{model_type}'")

        # Autoencoder para anomalías
        self.autoencoder: SensorAutoencoder | None = None
        if use_autoencoder:
            self.autoencoder = SensorAutoencoder(input_dim=input_dim)

        # Estadísticas para normalizar el score de anomalía
        self._anomaly_mean: float = 0.0
        self._anomaly_std: float = 1.0

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Crea secuencias con ventana deslizante.

        Args:
            X: Datos de forma (n_samples, input_dim).
            y: Etiquetas de forma (n_samples,). Se toma el valor del último paso.

        Returns:
            Tupla (secuencias, etiquetas) donde secuencias tiene forma
            (n_sequences, lookback, input_dim).
        """
        sequences: list[np.ndarray] = []
        labels: list[float] = []

        for i in range(len(X) - self.lookback + 1):
            seq = X[i : i + self.lookback]
            sequences.append(seq)
            if y is not None:
                labels.append(float(y[i + self.lookback - 1]))

        X_seq = np.array(sequences, dtype=np.float32)
        y_seq = np.array(labels, dtype=np.float32) if y is not None else None
        return X_seq, y_seq

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        val_split: float = 0.15,
    ) -> dict[str, list[float]]:
        """
        Entrena el clasificador completo.

        Args:
            X: Características de forma (n_samples, input_dim).
            y: Etiquetas binarias (0 = normal, 1 = brote).
            epochs: Épocas para el detector temporal.
            lr: Tasa de aprendizaje.
            batch_size: Tamaño de batch.
            val_split: Fracción de datos para validación.

        Returns:
            Diccionario con historial de entrenamiento.
        """
        # Crear secuencias
        X_seq, y_seq = self._create_sequences(X, y)

        # División cronológica
        n = len(X_seq)
        split_idx = int(n * (1 - val_split))

        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

        # DataLoaders
        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Entrenar detector temporal
        print(f"=== Entrenando detector {self.model_type.upper()} ===")
        history = self.detector.train_model(
            train_loader, val_loader, epochs=epochs, lr=lr
        )

        # Entrenar autoencoder con datos normales
        if self.use_autoencoder and self.autoencoder is not None:
            print("\n=== Entrenando Autoencoder ===")
            # Filtrar solo muestras normales (y == 0)
            normal_mask = y == 0
            X_normal = X[normal_mask]

            if len(X_normal) > 0:
                ae_ds = TensorDataset(torch.from_numpy(X_normal).float())
                ae_loader = DataLoader(ae_ds, batch_size=batch_size, shuffle=True)
                self.autoencoder.train_model(ae_loader, epochs=min(100, epochs * 2), lr=lr)

                # Establecer umbral de anomalía
                self.autoencoder.set_threshold(X_normal, percentile=95)

                # Calcular estadísticas de normalización
                errors = self.autoencoder.compute_reconstruction_error(X_normal)
                self._anomaly_mean = float(np.mean(errors))
                self._anomaly_std = float(np.std(errors) + 1e-8)

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la probabilidad de brote.

        Si ensemble está activo, combina:
            prob_final = 0.5 * prob_temporal + 0.5 * (1 - anomaly_score_normalized)

        donde anomaly_score_normalized = (error - mean) / std, acotado a [0, 1].

        Args:
            X: Características de forma (n_samples, input_dim).

        Returns:
            Array de probabilidades en [0, 1].
        """
        X_seq, _ = self._create_sequences(X)

        # Probabilidad del detector temporal
        temporal_prob = self.detector.predict(X_seq)

        if not self.ensemble or self.autoencoder is None:
            return temporal_prob

        # Score de anomalía
        # Usar la última observación de cada secuencia para el autoencoder
        last_points = X_seq[:, -1, :]  # (n_sequences, input_dim)
        errors = self.autoencoder.compute_reconstruction_error(last_points)

        # Normalizar: z-score → [0, 1] via sigmoid
        z = (errors - self._anomaly_mean) / self._anomaly_std
        anomaly_score = 1.0 / (1.0 + np.exp(-z))  # sigmoid → [0, 1]

        # Ensemble: promedio entre prob temporal y (1 - anomaly_score)
        final_prob = 0.5 * temporal_prob + 0.5 * (1.0 - anomaly_score)

        # Asegurar rango [0, 1]
        return np.clip(final_prob, 0.0, 1.0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """
        Evalúa el modelo y calcula métricas.

        Métricas calculadas:
            - AUC: Área bajo la curva ROC (roc_auc_score de sklearn).
            - F1: F1-score con umbral 0.5.
            - Precision@k: Precisión en el top 10% de predicciones.

        Args:
            X: Características.
            y: Etiquetas verdaderas.

        Returns:
            Diccionario con métricas {'auc': ..., 'f1': ..., 'precision_at_k': ...}.
        """
        from sklearn.metrics import f1_score, roc_auc_score

        # Obtener probabilidades
        proba = self.predict_proba(X)

        # Crear secuencias para alinear con y
        _, y_seq = self._create_sequences(X, y)

        if len(proba) != len(y_seq):
            # En caso de desalineación, usar el mínimo común
            min_len = min(len(proba), len(y_seq))
            proba = proba[:min_len]
            y_seq = y_seq[:min_len]

        # AUC
        try:
            auc = float(roc_auc_score(y_seq, proba))
        except ValueError:
            # Si solo hay una clase, AUC no está definido
            auc = 0.0

        # F1 con umbral 0.5
        y_pred = (proba >= 0.5).astype(int)
        f1 = float(f1_score(y_seq, y_pred, zero_division=0))

        # Precision@k: top 10% de predicciones
        k = max(1, int(len(proba) * 0.1))
        top_indices = np.argsort(proba)[::-1][:k]
        precision_at_k = float(np.mean(y_seq[top_indices]))

        return {
            "auc": auc,
            "f1": f1,
            "precision_at_k": precision_at_k,
        }

    def save(self, path: str) -> None:
        """
        Guarda el clasificador completo.

        Args:
            path: Directorio donde guardar los modelos.
        """
        os.makedirs(path, exist_ok=True)

        # Guardar detector temporal
        detector_path = os.path.join(path, f"detector_{self.model_type}.pt")
        self.detector.save_model(detector_path)

        # Guardar autoencoder
        if self.use_autoencoder and self.autoencoder is not None:
            ae_path = os.path.join(path, "autoencoder.pt")
            self.autoencoder.save_model(ae_path)

        # Guardar metadatos
        import json

        metadata = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "lookback": self.lookback,
            "use_autoencoder": self.use_autoencoder,
            "ensemble": self.ensemble,
            "anomaly_mean": self._anomaly_mean,
            "anomaly_std": self._anomaly_std,
        }
        meta_path = os.path.join(path, "classifier_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=True)

    def load(self, path: str) -> "OutbreakClassifier":
        """
        Carga el clasificador desde disco.

        Args:
            path: Directorio donde están guardados los modelos.

        Returns:
            self con pesos cargados.
        """
        import json

        # Cargar metadatos
        meta_path = os.path.join(path, "classifier_metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.model_type = metadata["model_type"]
        self.input_dim = metadata["input_dim"]
        self.lookback = metadata["lookback"]
        self.use_autoencoder = metadata["use_autoencoder"]
        self.ensemble = metadata["ensemble"]
        self._anomaly_mean = metadata["anomaly_mean"]
        self._anomaly_std = metadata["anomaly_std"]

        # Cargar detector temporal
        detector_path = os.path.join(path, f"detector_{self.model_type}.pt")
        if self.model_type == "lstm":
            self.detector = LSTMOutbreakDetector.load_model(detector_path)
        else:
            self.detector = TransformerOutbreakDetector.load_model(detector_path)

        # Cargar autoencoder
        if self.use_autoencoder:
            ae_path = os.path.join(path, "autoencoder.pt")
            self.autoencoder = SensorAutoencoder.load_model(ae_path)

        return self
