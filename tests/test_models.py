"""
Tests para los modelos de detección de brotes en camaroneras.

Ejecutar con:
    pytest tests/test_models.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.lstm_detector import LSTMOutbreakDetector
from src.models.transformer_detector import TransformerOutbreakDetector
from src.models.autoencoder_anomaly import SensorAutoencoder
from src.models.outbreak_classifier import OutbreakClassifier


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def dummy_input():
    """Input simulado: batch=4, seq_len=96, features=6."""
    return torch.randn(4, 96, 6)


@pytest.fixture
def dummy_input_2d():
    """Input 2D simulado para el autoencoder: batch=32, features=6."""
    return torch.randn(32, 6)


@pytest.fixture
def dummy_data():
    """Datos sintéticos para el clasificador."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(0, 1, (n, 6)).astype(np.float32)
    y = (rng.random(n) > 0.8).astype(np.float32)
    return X, y


# ===========================================================================
# Tests LSTM
# ===========================================================================
class TestLSTMOutbreakDetector:
    """Tests para el detector LSTM."""

    def test_forward_pass(self, dummy_input):
        """Verifica que el forward pass produzca output de forma correcta."""
        model = LSTMOutbreakDetector(input_dim=6, hidden_dims=[128, 64], dropout=0.3)
        output = model(dummy_input)
        assert output.shape == (4,), f"Esperaba (4,), obtuvo {output.shape}"

    def test_output_range(self, dummy_input):
        """Verifica que la salida esté en [0, 1]."""
        model = LSTMOutbreakDetector(input_dim=6)
        output = model(dummy_input)
        assert (output >= 0).all() and (output <= 1).all(), "Salida fuera de [0, 1]"

    def test_predict_returns_numpy(self, dummy_input):
        """Verifica que predict() retorne un array de numpy."""
        model = LSTMOutbreakDetector(input_dim=6)
        X = dummy_input.numpy()
        result = model.predict(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_save_load_roundtrip(self, dummy_input, tmp_path):
        """Verifica que save/load produzca resultados idénticos."""
        model = LSTMOutbreakDetector(input_dim=6, hidden_dims=[128, 64], dropout=0.3)
        model.eval()
        original_output = model(dummy_input)

        path = str(tmp_path / "lstm_model.pt")
        model.save_model(path)

        loaded_model = LSTMOutbreakDetector.load_model(path)
        loaded_model.eval()
        loaded_output = loaded_model(dummy_input)

        assert torch.allclose(original_output, loaded_output, atol=1e-6)


# ===========================================================================
# Tests Transformer
# ===========================================================================
class TestTransformerOutbreakDetector:
    """Tests para el detector Transformer."""

    def test_forward_pass(self, dummy_input):
        """Verifica que el forward pass produzca output de forma correcta."""
        model = TransformerOutbreakDetector(
            input_dim=6, d_model=64, n_heads=4, num_layers=2, dim_feedforward=256
        )
        output = model(dummy_input)
        assert output.shape == (4,), f"Esperaba (4,), obtuvo {output.shape}"

    def test_output_range(self, dummy_input):
        """Verifica que la salida esté en [0, 1]."""
        model = TransformerOutbreakDetector(input_dim=6)
        output = model(dummy_input)
        assert (output >= 0).all() and (output <= 1).all(), "Salida fuera de [0, 1]"

    def test_predict_returns_numpy(self, dummy_input):
        """Verifica que predict() retorne un array de numpy."""
        model = TransformerOutbreakDetector(input_dim=6)
        X = dummy_input.numpy()
        result = model.predict(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_save_load_roundtrip(self, dummy_input, tmp_path):
        """Verifica que save/load produzca resultados idénticos."""
        model = TransformerOutbreakDetector(input_dim=6, d_model=64, n_heads=4)
        model.eval()
        original_output = model(dummy_input)

        path = str(tmp_path / "transformer_model.pt")
        model.save_model(path)

        loaded_model = TransformerOutbreakDetector.load_model(path)
        loaded_model.eval()
        loaded_output = loaded_model(dummy_input)

        assert torch.allclose(original_output, loaded_output, atol=1e-6)


# ===========================================================================
# Tests Autoencoder
# ===========================================================================
class TestSensorAutoencoder:
    """Tests para el autoencoder de anomalías."""

    def test_forward_pass(self, dummy_input_2d):
        """Verifica que el forward pass produzca reconstrucción de forma correcta."""
        model = SensorAutoencoder(input_dim=6)
        output = model(dummy_input_2d)
        assert output.shape == dummy_input_2d.shape

    def test_reconstruction_error_shape(self, dummy_input_2d):
        """Verifica que compute_reconstruction_error retorne shape correcto."""
        model = SensorAutoencoder(input_dim=6)
        errors = model.compute_reconstruction_error(dummy_input_2d)
        assert errors.shape == (32,), f"Esperaba (32,), obtuvo {errors.shape}"

    def test_reconstruction_error_nonnegative(self, dummy_input_2d):
        """Verifica que los errores de reconstrucción sean no negativos."""
        model = SensorAutoencoder(input_dim=6)
        errors = model.compute_reconstruction_error(dummy_input_2d)
        assert (errors >= 0).all(), "Errores de reconstrucción negativos"

    def test_set_threshold(self, dummy_input_2d):
        """Verifica que set_threshold establezca un umbral válido."""
        model = SensorAutoencoder(input_dim=6)
        threshold = model.set_threshold(dummy_input_2d, percentile=95)
        assert isinstance(threshold, float)
        assert threshold > 0
        assert model.threshold == threshold

    def test_detect_anomalies(self, dummy_input_2d):
        """Verifica que detect_anomalies retorne array booleano."""
        model = SensorAutoencoder(input_dim=6)
        model.set_threshold(dummy_input_2d, percentile=50)
        anomalies = model.detect_anomalies(dummy_input_2d)
        assert anomalies.dtype == bool
        assert anomalies.shape == (32,)

    def test_detect_anomalies_no_threshold_raises(self, dummy_input_2d):
        """Verifica que detect_anomalies lance error sin umbral."""
        model = SensorAutoencoder(input_dim=6)
        with pytest.raises(ValueError, match="umbral"):
            model.detect_anomalies(dummy_input_2d)

    def test_save_load_roundtrip(self, dummy_input_2d, tmp_path):
        """Verifica que save/load produzca resultados idénticos."""
        model = SensorAutoencoder(input_dim=6)
        model.set_threshold(dummy_input_2d, percentile=95)
        original_errors = model.compute_reconstruction_error(dummy_input_2d)

        path = str(tmp_path / "autoencoder.pt")
        model.save_model(path)

        loaded_model = SensorAutoencoder.load_model(path)
        loaded_errors = loaded_model.compute_reconstruction_error(dummy_input_2d)

        assert np.allclose(original_errors, loaded_errors, atol=1e-6)
        assert loaded_model.threshold == model.threshold

    def test_3d_input(self):
        """Verifica que el autoencoder funcione con entrada 3D."""
        model = SensorAutoencoder(input_dim=6)
        X = torch.randn(8, 96, 6)
        output = model(X)
        assert output.shape == X.shape

        errors = model.compute_reconstruction_error(X)
        assert errors.shape == (8,)


# ===========================================================================
# Tests OutbreakClassifier
# ===========================================================================
class TestOutbreakClassifier:
    """Tests para el clasificador de brotes."""

    def test_predict_proba_range(self, dummy_data):
        """Verifica que predict_proba retorne valores en [0, 1]."""
        X, y = dummy_data
        # Usar lookback pequeño para que haya suficientes secuencias con datos dummy
        clf = OutbreakClassifier(
            model_type="lstm", input_dim=6, lookback=10,
            use_autoencoder=True, ensemble=True,
        )
        clf.train(X, y, epochs=2, batch_size=16)
        proba = clf.predict_proba(X)
        assert isinstance(proba, np.ndarray)
        assert (proba >= 0).all() and (proba <= 1).all(), "Probabilidades fuera de [0, 1]"

    def test_predict_proba_shape(self, dummy_data):
        """Verifica que predict_proba retorne el shape correcto."""
        X, y = dummy_data
        clf = OutbreakClassifier(
            model_type="lstm", input_dim=6, lookback=10,
            use_autoencoder=False, ensemble=False,
        )
        clf.train(X, y, epochs=1, batch_size=16)
        proba = clf.predict_proba(X)
        expected_len = max(0, len(X) - 10 + 1)
        assert len(proba) == expected_len

    def test_evaluate_returns_dict(self, dummy_data):
        """Verifica que evaluate retorne diccionario con métricas correctas."""
        X, y = dummy_data
        clf = OutbreakClassifier(
            model_type="lstm", input_dim=6, lookback=10,
            use_autoencoder=False, ensemble=False,
        )
        clf.train(X, y, epochs=2, batch_size=16)
        metrics = clf.evaluate(X, y)
        assert isinstance(metrics, dict)
        assert "auc" in metrics
        assert "f1" in metrics
        assert "precision_at_k" in metrics
        assert 0.0 <= metrics["auc"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["precision_at_k"] <= 1.0

    def test_transformer_classifier(self, dummy_data):
        """Verifica que el clasificador funcione con Transformer."""
        X, y = dummy_data
        clf = OutbreakClassifier(
            model_type="transformer", input_dim=6, lookback=10,
            use_autoencoder=False, ensemble=False,
        )
        clf.train(X, y, epochs=2, batch_size=16)
        proba = clf.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_invalid_model_type_raises(self):
        """Verifica que model_type inválido lance error."""
        with pytest.raises(ValueError, match="model_type"):
            OutbreakClassifier(model_type="invalid")

    def test_save_load_roundtrip(self, dummy_data, tmp_path):
        """Verifica que save/load del clasificador funcione."""
        X, y = dummy_data
        clf = OutbreakClassifier(
            model_type="lstm", input_dim=6, lookback=10,
            use_autoencoder=True, ensemble=True,
        )
        clf.train(X, y, epochs=2, batch_size=16)

        # Predecir antes de guardar
        proba_before = clf.predict_proba(X)

        # Guardar y recargar
        save_dir = str(tmp_path / "classifier")
        clf.save(save_dir)

        loaded_clf = OutbreakClassifier(
            model_type="lstm", input_dim=6, lookback=10,
            use_autoencoder=True, ensemble=True,
        )
        loaded_clf.load(save_dir)

        # Predecir después de cargar
        proba_after = loaded_clf.predict_proba(X)

        assert np.allclose(proba_before, proba_after, atol=1e-5), (
            "Las predicciones no coinciden después de save/load"
        )
