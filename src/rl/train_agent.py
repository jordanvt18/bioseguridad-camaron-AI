"""
Script de entrenamiento del agente de aprendizaje por refuerzo.

Entrena un agente PPO o DQN sobre el entorno ShrimpPondEnv usando
Stable Baselines3. Incluye callbacks de checkpoint y evaluación.

Uso:
    python -m src.rl.train_agent --algorithm ppo --timesteps 100000 --lr 3e-4

    python -m src.rl.train_agent --algorithm dqn --timesteps 50000 --lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

# Asegurar que el paquete sea importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rl.environment import ShrimpPondEnv


def check_sb3_available() -> bool:
    """Verifica si Stable Baselines3 está instalado."""
    try:
        import stable_baselines3  # noqa: F401
        return True
    except ImportError:
        return False


def train_agent(
    algorithm: str = "ppo",
    timesteps: int = 100_000,
    learning_rate: float = 3e-4,
    model_save_path: str = "models",
    checkpoint_freq: int = 10_000,
    eval_freq: int = 5_000,
    n_eval_episodes: int = 20,
) -> Optional[str]:
    """
    Entrena un agente RL sobre ShrimpPondEnv.

    Args:
        algorithm: Algoritmo a usar ('ppo' o 'dqn').
        timesteps: Número de pasos de entrenamiento.
        learning_rate: Tasa de aprendizaje.
        model_save_path: Directorio para guardar el modelo.
        checkpoint_freq: Frecuencia de guardado de checkpoint (pasos).
        eval_freq: Frecuencia de evaluación (pasos).
        n_eval_episodes: Número de episodios de evaluación.

    Returns:
        Ruta al modelo guardado, o None si falló el entrenamiento.
    """
    if not check_sb3_available():
        print("=" * 60)
        print("ERROR: Stable Baselines3 no está instalado.")
        print("=" * 60)
        print("\nPara instalarlo, ejecute:")
        print("  pip install stable-baselines3[extra] gymnasium")
        print("\nO con conda:")
        print("  conda install -c conda-forge stable-baselines3 gymnasium")
        print("\nUna vez instalado, vuelva a ejecutar este script.")
        return None

    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Crear entorno
    env = ShrimpPondEnv()
    eval_env = ShrimpPondEnv()

    # Vectorizar entorno
    vec_env = DummyVecEnv([lambda: env])

    # Configurar algoritmo
    algorithm = algorithm.lower().strip()
    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./logs/rl/",
        )
        model_name = "ppo_shrimp"
    elif algorithm == "dqn":
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            buffer_size=50_000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./logs/rl/",
        )
        model_name = "dqn_shrimp"
    else:
        print(f"Algoritmo no soportado: {algorithm}. Use 'ppo' o 'dqn'.")
        return None

    # Crear directorio de modelos
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(os.path.join(model_save_path, "checkpoints"), exist_ok=True)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // 1, 1),
        save_path=os.path.join(model_save_path, "checkpoints"),
        name_prefix=model_name,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_path, "best"),
        log_path=os.path.join(model_save_path, "eval_logs"),
        eval_freq=max(eval_freq // 1, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    # Entrenar
    print("=" * 60)
    print(f"  Entrenando agente {algorithm.upper()}")
    print(f"  Pasos: {timesteps:,}")
    print(f"  Tasa de aprendizaje: {learning_rate}")
    print(f"  Checkpoint cada: {checkpoint_freq:,} pasos")
    print(f"  Evaluación cada: {eval_freq:,} pasos")
    print("=" * 60)

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=False,
    )

    # Guardar modelo final
    model_path = os.path.join(model_save_path, f"{model_name}.zip")
    model.save(model_path)
    print(f"\nModelo guardado en: {model_path}")

    # Evaluar agente entrenado
    print("\n" + "=" * 60)
    print("  Evaluación del agente entrenado")
    print("=" * 60)

    rewards = []
    for ep in range(n_eval_episodes):
        obs, _ = eval_env.reset(seed=ep)
        ep_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)
        print(f"  Episodio {ep + 1:3d}/{n_eval_episodes}: Recompensa = {ep_reward:.2f}")

    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    print(f"\n  Recompensa promedio: {avg_reward:.2f} ± {std_reward:.2f}")
    print("=" * 60)

    return model_path


def main() -> None:
    """Función principal con parsing de argumentos."""
    parser = argparse.ArgumentParser(
        description="Entrena un agente RL para políticas de bioseguridad camaronera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m src.rl.train_agent --algorithm ppo --timesteps 100000
  python -m src.rl.train_agent --algorithm dqn --timesteps 50000 --lr 0.001
  python -m src.rl.train_agent --timesteps 200000 --lr 3e-4 --checkpoint 5000
        """,
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="Algoritmo de RL a usar (por defecto: ppo)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Número de pasos de entrenamiento (por defecto: 100000)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=3e-4,
        dest="learning_rate",
        help="Tasa de aprendizaje (por defecto: 3e-4)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=10_000,
        dest="checkpoint_freq",
        help="Frecuencia de checkpoint en pasos (por defecto: 10000)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        dest="eval_freq",
        help="Frecuencia de evaluación en pasos (por defecto: 5000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        dest="n_eval_episodes",
        help="Número de episodios de evaluación (por defecto: 20)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models",
        dest="model_save_path",
        help="Directorio para guardar el modelo (por defecto: models)",
    )

    args = parser.parse_args()

    train_agent(
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        model_save_path=args.model_save_path,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
    )


if __name__ == "__main__":
    main()
