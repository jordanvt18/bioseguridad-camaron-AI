"""
Detector de brotes de enfermedad en camarones basado en LSTM.

Arquitectura:
    - Capa de entrada: 6 características de sensores
    - 2 capas LSTM (128, 64 unidades ocultas)
    - Dropout 0.3
    - Capa completamente conectada → sigmoid (probabilidad 0-1)

Ventana de observación: 96 pasos temporales (24 horas a intervalos de 15 min).
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LSTMOutbreakDetector(nn.Module):
    """Detector de brotes usando una red LSTM bidireccional."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        num_classes: int = 1,
    ) -> None:
        """
        Inicializa el detector LSTM.

        Args:
            input_dim: Número de características de entrada (sensores).
            hidden_dims: Lista con unidades ocultas por capa LSTM.
            dropout: Tasa de dropout entre capas.
            num_classes: Número de clases de salida (1 = probabilidad binaria).
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.num_classes = num_classes

        # Construir capas LSTM dinámicamente
        layers: list[nn.Module] = []
        prev_hidden = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(
                nn.LSTM(
                    input_size=prev_hidden,
                    hidden_size=h,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                )
            )
            layers.append(nn.Dropout(dropout))
            prev_hidden = h

        self.lstm_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i, h in enumerate(hidden_dims):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_dim if i == 0 else hidden_dims[i - 1],
                    hidden_size=h,
                    num_layers=1,
                    batch_first=True,
                )
            )
            self.dropouts.append(nn.Dropout(dropout))

        # Capa de salida
        self.fc = nn.Linear(hidden_dims[-1], num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de forma (batch, seq_len, input_dim).

        Returns:
            Tensor de forma (batch,) con probabilidades en [0, 1].
        """
        out = x
        for i, (lstm, drop) in enumerate(zip(self.lstm_layers, self.dropouts)):
            out, _ = lstm(out)
            out = drop(out)

        # Tomar el último paso temporal
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze(-1)

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
            device: Dispositivo ('cuda' o 'cpu'). Auto-detección si es None.

        Returns:
            Diccionario con historial de pérdidas {'train': [...], 'val': [...]}.
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

        # Restaurar mejores pesos
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
            Array de probabilidades en [0, 1] con forma (n_samples,).
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
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "num_classes": self.num_classes,
            },
            path,
        )

    @classmethod
    def load_model(cls, path: str, device: str | None = None) -> "LSTMOutbreakDetector":
        """
        Carga un modelo desde disco.

        Args:
            path: Ruta del archivo guardado.
            device: Dispositivo destino.

        Returns:
            Instancia de LSTMOutbreakDetector con pesos cargados.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
            num_classes=checkpoint["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model
