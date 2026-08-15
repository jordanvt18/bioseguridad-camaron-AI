"""
Autoencoder para detección de anomalías en sensores de estanques camaroneros.

Arquitectura:
    - Encoder: 6 → 32 → 16 → 8 (ReLU)
    - Decoder: 8 → 16 → 32 → 6 (ReLU)

Una anomalía se detecta cuando el error de reconstrucción supera un umbral
calculado como el percentil 95 de los errores sobre datos normales.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SensorAutoencoder(nn.Module):
    """Autoencoder denso para detección de anomalías en sensores."""

    def __init__(self, input_dim: int = 6) -> None:
        """
        Inicializa el autoencoder.

        Args:
            input_dim: Número de características de entrada (sensores).
        """
        super().__init__()
        self.input_dim = input_dim

        # Encoder: input_dim → 32 → 16 → 8
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        # Decoder: 8 → 16 → 32 → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.ReLU(),
        )

        # Umbral de anomalía (se calcula con set_threshold)
        self.threshold: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode → decode.

        Args:
            x: Tensor de forma (batch, input_dim) o (batch, seq_len, input_dim).

        Returns:
            Reconstrucción de la misma forma que x.
        """
        original_shape = x.shape
        if x.dim() == 3:
            # Aplanar para el autoencoder denso
            x = x.reshape(-1, original_shape[-1])

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Restaurar forma original
        if len(original_shape) == 3:
            decoded = decoded.reshape(original_shape)

        return decoded

    def train_model(
        self,
        data_loader: DataLoader,
        epochs: int = 100,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> list[float]:
        """
        Entrena el autoencoder para reconstruir datos normales.

        Args:
            data_loader: DataLoader con datos normales (sin anomalías).
            epochs: Número de épocas.
            lr: Tasa de aprendizaje.
            device: Dispositivo ('cuda' o 'cpu').

        Returns:
            Lista de pérdidas por época.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses: list[float] = []

        for epoch in range(epochs):
            self.train()
            epoch_losses: list[float] = []
            for batch in data_loader:
                # El batch puede ser (x,) o (x, y)
                if isinstance(batch, (tuple, list)):
                    batch_x = batch[0]
                else:
                    batch_x = batch
                batch_x = batch_x.float().to(device)

                optimizer.zero_grad()
                reconstructed = self(batch_x)

                # Si el batch tiene forma (batch, seq_len, input_dim),
                # aplanamos para calcular la pérdida
                if batch_x.dim() == 3:
                    target = batch_x.reshape(-1, batch_x.shape[-1])
                    pred = reconstructed.reshape(-1, batch_x.shape[-1])
                else:
                    target = batch_x
                    pred = reconstructed

                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses))
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Época {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

        return losses

    @torch.no_grad()
    def compute_reconstruction_error(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Calcula el error de reconstrucción por muestra.

        Args:
            X: Datos de forma (n_samples, input_dim) o (n_samples, seq_len, input_dim).

        Returns:
            Array de errores con forma (n_samples,). Si la entrada es 3D,
            se promedia el error sobre la dimensión temporal.
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(device)

        reconstructed = self(X)

        # Error MSE por característica, luego promediar
        if X.dim() == 3:
            # (n, seq_len, input_dim) → error por muestra
            errors = torch.mean((X - reconstructed) ** 2, dim=[1, 2])
        else:
            # (n, input_dim) → error por muestra
            errors = torch.mean((X - reconstructed) ** 2, dim=1)

        return errors.cpu().numpy()

    def detect_anomalies(
        self,
        X: np.ndarray | torch.Tensor,
        threshold: float | None = None,
    ) -> np.ndarray:
        """
        Detecta anomalías basadas en el error de reconstrucción.

        Args:
            X: Datos de entrada.
            threshold: Umbral de error. Si es None, usa self.threshold.

        Returns:
            Array booleano: True si la muestra es anomalía.
        """
        if threshold is None:
            threshold = self.threshold
        if threshold is None:
            raise ValueError(
                "No se ha definido un umbral. Llama set_threshold() primero "
                "o proporciona un threshold explícito."
            )

        errors = self.compute_reconstruction_error(X)
        return errors > threshold

    @torch.no_grad()
    def set_threshold(
        self,
        X_normal: np.ndarray | torch.Tensor,
        percentile: float = 95.0,
    ) -> float:
        """
        Calcula y establece el umbral de anomalía basado en datos normales.

        Args:
            X_normal: Datos considerados normales.
            percentile: Percentil de la distribución de errores a usar como umbral.

        Returns:
            El umbral calculado.
        """
        errors = self.compute_reconstruction_error(X_normal)
        self.threshold = float(np.percentile(errors, percentile))
        print(
            f"Umbral establecido en {self.threshold:.6f} "
            f"(percentil {percentile} de errores normales)."
        )
        return self.threshold

    def save_model(self, path: str) -> None:
        """
        Guarda el autoencoder en disco.

        Args:
            path: Ruta del archivo (.pt o .pth).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load_model(cls, path: str, device: str | None = None) -> "SensorAutoencoder":
        """
        Carga un autoencoder desde disco.

        Args:
            path: Ruta del archivo guardado.
            device: Dispositivo destino.

        Returns:
            Instancia de SensorAutoencoder con pesos cargados.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(input_dim=checkpoint["input_dim"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.threshold = checkpoint.get("threshold", None)
        model.to(device)
        model.eval()
        return model
