"""
This script is a custom Gymnasium wrapper for the single-qubit Schrodinger evolutiona s a bang-bang control MDP

The MDP (copied from Sec. II of Bukov et al.):
    state (observation): (normalised time step, current field / h_max)

    action : discrete {0, 1} -> set field to {-h_max, +h_max} (pure bang-bang; the agent has no h = 0 option)
                           
    transition: evolve |psi> by exp(-i H[h] dt)

    reward: 0 for every step except the last, where it is the final fidelity |<psi_*|psi(T)>|^2  

    episode length : N_T fixed steps, dt = T / N_T

"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from physics import PSI_I, PSI_TARGET, evolve, fidelity


class QubitEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, T=2.4, n_steps=24, h_max=4.0,
                 psi_i=PSI_I, psi_target=PSI_TARGET):
        super().__init__()
        self.T = float(T)
        self.n_steps = int(n_steps)
        self.dt = self.T / self.n_steps
        self.h_max = float(h_max)
        self.psi_i = psi_i.astype(complex)
        self.psi_target = psi_target.astype(complex)

        # two bang-bang actions, 0: h = -h_max, 1: h = +h_max
        self.action_space = spaces.Discrete(2)
        # observation: [step / n_steps, current_field / h_max]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self.psi = None
        self.step_idx = None
        self.h_current = None
        self.protocol = None  # for plotting the final protocol after an episode

    def _obs(self):
        return np.array(
            [self.step_idx / self.n_steps, self.h_current / self.h_max],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.psi = self.psi_i.copy()
        self.step_idx = 0
        self.h_current = 0.0
        self.protocol = []
        return self._obs(), {}

    def step(self, action):
        if int(action) == 0:
            h = -self.h_max
        else:
            h = self.h_max

        self.h_current = h
        self.protocol.append(h)

        self.psi = evolve(self.psi, h, self.dt) # evolve one step under the chosen field
        self.step_idx += 1

        if self.step_idx >= self.n_steps:
            terminated = True
        else:
            terminated = False
            
        if terminated:
            reward = fidelity(self.psi, self.psi_target)
        else:
            reward = 0.0
        truncated = False
        info = {"fidelity": fidelity(self.psi, self.psi_target)} if terminated else {}
        return self._obs(), float(reward), terminated, truncated, info

