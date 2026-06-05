"""
We compute the ground truth, since by keeping the system simple it is tractable.
We use it to benchmark the agents results.

Method: stochastic descent (SD) - greedy 1-flip local search from random starts.
This is the SD algorithm Bukov uses to find the bang-bang optimum F_h(T), and it
scales to any N_T (unlike brute force, which is 2^N_T and only works for tiny N_T).
"""

import numpy as np
from scipy.linalg import expm
from physics import hamiltonian, PSI_I, PSI_TARGET, fidelity


def _fidelity_of_protocol(bits, U_minus, U_plus):
    """
    Final-state fidelity of a bang-bang protocol (1 -> +h_max, 0 -> -h_max).

    U_minus and U_plus are the precomputed one-step propagators for the two field
    values. There are only two possible fields at a fixed dt, so we build those two
    matrices once (in stochastic_descent) and just multiply here
    """
    psi = PSI_I.copy()
    for b in bits:
        psi = (U_plus if b else U_minus) @ psi
    return fidelity(psi, PSI_TARGET)


def stochastic_descent(T, n_steps, h_max=4.0, n_restarts=20, seed=0):
    """
    A reimplementation of the stochastic descent (SD) algorithm from Bukov's paper,
    using bit flips instead of swaps.

    """
    rng = np.random.default_rng(seed)

    # the only two propagators we ever need, built once for this (T, n_steps, h_max)
    dt = T / n_steps
    U_minus = expm(-1j * hamiltonian(-h_max) * dt)
    U_plus = expm(-1j * hamiltonian(+h_max) * dt)

    best_F, best_bits = 0.0, None
    for _ in range(n_restarts):

        bits = rng.integers(0, 2, size=n_steps).tolist() #sample random startign protocol
        F = _fidelity_of_protocol(bits, U_minus, U_plus) # compute its fidelity
        improved = True
        while improved: # 1-flip greedy local search
            improved = False
            order = rng.permutation(n_steps) # random order of bits to try flipping, to avoid bias
            for i in order:
                bits[i] ^= 1 #flip bit i by xoring with 1
                F_try = _fidelity_of_protocol(bits, U_minus, U_plus) #compute new fidelity
                if F_try > F:
                    F, improved = F_try, True # if we improve, store and continue
                else:
                    bits[i] ^= 1 #if not improved revert
        if F > best_F:
            best_F, best_bits = F, bits[:] #store the best protocol found so far
    fields = [h_max if b else -h_max for b in best_bits] #convert bits to fields for easier comparison with agents
    return best_F, fields


if __name__ == "__main__":
    # quick look at the optimum across the three phases
    for T in [0.3, 1.0, 2.4, 3.0]:
        Fs, _ = stochastic_descent(T, n_steps=28, n_restarts=30)
        print(f"  T={T:>4}: SD optimum = {Fs:.4f}")
