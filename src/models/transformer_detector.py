"""
Detector de brotes de enfermedad en camarones basado en Transformer.

Arquitectura:
    - Codificador posicional sinusoidal
    - Transformer Encoder: 4 cabezas de atención, 2 capas, dim_feedforward=256
    - Capa completamente conectada → sigmoid (probabilidad 0-1)

Ventana de observación: 96 pasos temporales (24 horas a intervalos de 15 min).
"""

from __future__ import annotations

import math
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PositionalEncoding(nn.Module):
    """Codificación posicional sinusoidal para secuencias temporales."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        """
        Args:
            d_model: Dimensión del embedding.
            max_len: Longitud máxima de secuencia.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forma (batch, seq_len, d_model).
        Returns:
            Tensor con codificación posicional añadida.
        """
        return x + self.pe[:, : x.size(1), :]


class TransformerOutbreakDetector(nn.Module):
    """Detector de brotes usando un Transformer Encoder."""

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 1,
    ) -> None:
        """
        Inicializa el detector Transformer.

        Args:
            input_dim: Número de características de entrada.
            d_model: Dimensión interna del modelo.
            n_heads: Número de cabezas de atención.
            num_layers: Número de capas del encoder.
            dim_feedforward: Dimensión de la red feedforward interna.
            dropout: Tasa de dropout.
            num_classes: Número de clases de salida.
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.num_classes = num_classes

        # Proyección de entrada a d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de forma (batch, seq_len, input_dim).

        Returns:
            Tensor de forma (batch,) con probabilidades en [0, 1].
        """
        # Proyectar entrada a d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Pooling: usar el último paso temporal
        x = x[:, -1, :]  # (batch, d_model)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 10,
        device: str | None = None,
    ) -> dict[str, list[float]]:
        """
        Entrena el modelo con early stopping.

        Args:
            train_loader: DataLoader de entrenamiento.
            val_loader: DataLoader de validación.
            epochs: Número máximo de épocas.
            lr: Tasa de aprendizaje.
            patience: Paciencia para early stopping.
            device: Dispositivo ('cuda' o 'cpu').

        Returns:
            Diccionario con historial de pérdidas.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        history: dict[str, list[float]] = {"train": [], "val": []}

        for epoch in range(epochs):
            # --- Entrenamiento ---
            self.train()
            train_losses: list[float] = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # --- Validación ---
            self.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    outputs = self(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            history["train"].append(avg_train)
            history["val"].append(avg_val)

            print(
                f"Época {epoch + 1}/{epochs} - "
                f"Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}"
            )

            # Early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping en época {epoch + 1} (patiencia={patience}).")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    @torch.no_grad()
    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Predice probabilidad de brote para cada secuencia.

        Args:
            X: Array de forma (n_samples, seq_len, input_dim) o tensor.

        Returns:
            Array de probabilidades en [0, 1].
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(device)

        outputs = self(X)
        return outputs.cpu().numpy()

    def save_model(self, path: str) -> None:
        """
        Guarda el modelo en disco.

        Args:
            path: Ruta del archivo (.pt o .pth).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout_rate,
                "num_classes": self.num_classes,
            },
            path,
        )

    @classmethod
    def load_model(cls, path: str, device: str | None = None) -> "TransformerOutbreakDetector":
        """
        Carga un modelo desde disco.

        Args:
            path: Ruta del archivo guardado.
            device: Dispositivo destino.

        Returns:
            Instancia de TransformerOutbreakDetector con pesos cargados.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            input_dim=checkpoint["input_dim"],
            d_model=checkpoint["d_model"],
            n_heads=checkpoint["n_heads"],
            num_layers=checkpoint["num_layers"],
            dim_feedforward=checkpoint["dim_feedforward"],
            dropout=checkpoint["dropout"],
            num_classes=checkpoint["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model
