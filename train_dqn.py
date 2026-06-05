"""
Script to train the DQN agent to be compared with tabular Q-learning in q_learning.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from qubit_env import QubitEnv


class FidelityLogger(BaseCallback):
    """Stores the end-of-episode fidelity to plot the learning curve."""
    def __init__(self):
        super().__init__()
        self.fidelities = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "fidelity" in info:
                self.fidelities.append(info["fidelity"])
        return True


def train(T=2.4, n_steps=24, total_timesteps=150_000, seed=0):
    env = QubitEnv(T=T, n_steps=n_steps)

    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[64, 64])
    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=64,
        gamma=1.0,                 # undiscounted episodic task
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.6,
        exploration_final_eps=0.02,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )
    logger = FidelityLogger()
    model.learn(total_timesteps=total_timesteps, callback=logger)

    # greedy rollout
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(int(action))
    return model, env.protocol, info["fidelity"], np.array(logger.fidelities)


def plot_run(T, protocol, F, history, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # learning curve (red dots = per-episode, blue line = running mean)
    ax1.plot(history, ".", color="tab:red", ms=2, alpha=0.3)
    if len(history) > 50:
        kernel = np.ones(50) / 50
        run = np.convolve(history, kernel, mode="valid")
        ax1.plot(np.arange(len(run)) + 25, run, color="tab:blue", lw=2)
    ax1.set_xlabel("episode")
    ax1.set_ylabel("final fidelity")
    ax1.set_ylim(0, 1.02)
    ax1.set_title(f"DQN learning curve (T={T})")

    # learned protocol
    dt = T / len(protocol)
    t = np.arange(len(protocol)) * dt
    ax2.step(np.append(t, T), np.append(protocol, protocol[-1]),
             where="post", color="tab:red", lw=2)
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$h_x(t)$")
    ax2.set_ylim(-5, 5)
    ax2.set_title(f"learned protocol, F = {F:.3f}")

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    T = 2.4
    model, protocol, F, history = train(T=T, n_steps=24, total_timesteps=150_000)
    print(f"DQN final fidelity at T={T}: {F:.4f}")
    plot_run(T, protocol, F, history, "dqn_T2.4.png")
    print("Saved dqn_T2.4.png")
