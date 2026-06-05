"""
We make a simple environment for the single-qubit control problem. 

All the physics logic is handled by physics.py.

Observable, the actual state:
    Like in  Bukov et al., the agent never sees the quantum state, only the
    pair (time step, current field). In the bang-bang case the field is only ever
    -h_max or +h_max, so we have two field values. We need to store the state in
    as single integer so that pack so the agent can keep a plain 2D Q-table (state x action).
    We do so with this simple encoding:
        state = t * N_FIELDS + field_index          
        
    with field_index in {0, 1} corresponding to the two possible field values. 
"""
import numpy as np
from scipy.linalg import expm

from physics import hamiltonian, PSI_I, PSI_TARGET, fidelity

# bang-bang: the field is either -h_max or +h_max, so there are two field values
N_FIELDS = 2


class Environment:
    def __init__(self, T=2.4, n_steps=48, h_max=4.0):
        self.T = T
        self.n_steps = n_steps
        self.h_max = h_max
        self.dt = T / n_steps

        # action i sets the field to fields[i]
        self.fields = np.array([-h_max, +h_max])

        # We only evolve under h = -h_max or h = +h_max, so we precompute
        # the propagators once here instead of calling expm inside every single step
        self.propagators = [expm(-1j * hamiltonian(h) * self.dt) for h in self.fields]

        # how many distinct states the agent can be in (used to determine the size of the Q-table)
        self.n_states = self.n_steps * N_FIELDS

        self.reset()

    def encode(self, t, field_idx):
        """
        Encode the state as a single integer
        """
        return t * N_FIELDS + field_idx

    def reset(self):
        self.psi = PSI_I.copy() # restore to initial state
        self.t = 0
        self.field_idx = 0  # start at h = -h_max, same as the paper
        self.state = self.encode(self.t, self.field_idx) # encode the initial state as an integer
        return self.state

    def is_ended(self):
        return self.t >= self.n_steps

    def reward(self):
        # as in the paper we use a sparse reward that is zero everywhere except the very last step
        # where we return the final fidelity. This sparse rewars is the main source of difficulty 
        # for the agents, but to make it physically reaslitic that is what authors had to do.
        if self.is_ended():
            return fidelity(self.psi, PSI_TARGET)
        return 0.0

    def step(self, action):
        """
        Apply one field pulse for a time dt and return the Markov tuple
        (old_state, action, reward, new_state, done)

        """
        old_state = self.state

        # evolve the qubit under the field for one step
        self.psi = self.propagators[action] @ self.psi
        self.t += 1 # advance time by one step
        self.field_idx = action  # next fiels index is the action we just took

        # clamp the time index for the encoded "new state" so we never go out of the table
        t_for_state = min(self.t, self.n_steps - 1)
        self.state = self.encode(t_for_state, self.field_idx)

        return (old_state, action, self.reward(), self.state, self.is_ended())

