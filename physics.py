"""
Single-qubit physics logic for the RL quantum-control project

S_z, S_x are the spin-1/2 operators (Pauli matrices / 2). and are linked to the hamiltonian H(h) as follows:

H(h) = -S_z - h * S_x          (Eq. 3 of the paper)

The agent can only control the control field h(t). Since we are in a Bang-bang control setting, it can only choose from the set {+4, -4} at each time step.
Initial state |psi_i> = ground state of H(h = -2)
Target state |psi_*> = ground state of H(h = +2)

The reward will be computed by the final fidelity only at the last time step T that it will be varied to observe the different phases.

Reward = final fidelity  F = |<psi_*|psi(T)>|^2
"""
#imports 
import numpy as np
from scipy.linalg import expm #for computing the matrix exponential needed to evolve the state




# Pauli / spin-1/2 operators, needed to define the hamiltonian and evolve the state. They are 2x2 complex matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

Sx = 0.5 * sigma_x
Sy = 0.5 * sigma_y
Sz = 0.5 * sigma_z


def hamiltonian(h):
    """
    H(h) = -S_z - h S_x  (2x2 complex matrix)
    Its the hamiltonian of the system for a given control field h. It is used to compute the ground state and to evolve the state.
    """
    return -Sz - h * Sx


def ground_state(h):
    """
    Normalised ground state of H(h)
    Diagonalising the hamiltonian gives us the eigenvalues (energies) and eigenvectors (states). 
    The ground state is the one with the lowest energy, which is the first one in sorted order. 
    We return it normalised to ensure it has unit norm.
    """
    evals, evecs = np.linalg.eigh(hamiltonian(h))
    psi = evecs[:, 0] # eight returns in sorted order so the first will always be the ground state
    return psi / np.linalg.norm(psi)


def evolve(psi, h, dt):
    """
    Evolve psi one time step dt under the constant field h

    |psi_new> = exp(-i*H[h]*dt)|psi_current>
       
    """
    U = expm(-1j * hamiltonian(h) * dt) #time evolution operator, iits unitary.
    return U @ psi #trasform the state by applying U


def fidelity(psi, psi_target):
    """
    Compute the fidelity between current state and target state to use as reward
    F = |<psi_target | psi>|^2
    
    """
    return np.abs(np.vdot(psi_target, psi)) ** 2



# Default initial and target states, same as the paper
PSI_I = ground_state(-2.0)
PSI_TARGET = ground_state(+2.0)


if __name__ == "__main__":
    # debug print
    print("Initial state |psi_i> =", np.round(PSI_I, 3))
    print("Target  state |psi_*> =", np.round(PSI_TARGET, 3))
    print("Initial fidelity F(0) =", round(fidelity(PSI_I, PSI_TARGET), 4))
    # evolving with a constant field should preserve the norm
    psi = PSI_I.copy()
    for _ in range(50):
        psi = evolve(psi, 4.0, 0.05)
    print("Norm after 50 steps =", round(np.linalg.norm(psi), 6))
    # norm is preserved, all seems good!
