"""
Script to compare the different methods for the single qubit bang-bang control problem.
It draws the main figure of the paper, showing the final fidelity F_h(T) achieved by
different methods as a function of the protocol duration T, across the three phases.

The methods we compare are:

    - optimum  
            stochastic descent from benchmark.py. This is the reference
            curve everything else is measured against.

    - simple Q       
            plain one-step tabular Q-learning (train_simple), as baseline 
            to show that the problem is actually hard for a naive agent. 

    - Bukov Q(lambda)
            Bukov et al. method: traces + softmax ramp + replay + force-learn (train_bukov). 

    - DQN vanilla
            stable-baselines3 deep Q-network on the same (t, h) gym env,
            to see what off-the-shelf deep RL does here (not very well).

    - DQN smart
            the same blind DQN but given the two RL mechanisms it was
            missing (Monte-Carlo returns + best-episode replay). No physics
            knowledge added; it just gets honest credit assignment and a
            memory of its best protocol. We report the best protocol it found
            (like SD and the force-learned Bukov agent). Its greedy policy is
            often worse - the aliasing ceiling - which we explain in dqn_smart.py.

All five use exactly the same blind observation [t/N_T, h/h_max] and the same single
terminal reward. The optimum is cheap, so we draw it on a dense grid of T. The
learners are slower, so we only run them at a handful of T values across the phases.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from benchmark import stochastic_descent
from main_qubit import train_simple, train_bukov
from train_dqn import train as train_dqn
from dqn_smart import train_dqn_smart

# we fix the number of pulses for the comparison so every method gets the same action budget
#  only the total duration T changes from point to point
N_STEPS = 28
T_C, T_QSL = 0.618, 2.415 # the two phase boundaries from the paper

T_DENSE = np.linspace(0.1, 4.0, 25)  # for the optimum reference line
T_AGENTS = [0.3, 0.8, 1.2, 1.8, 2.4, 3.2]  # where we actually run the learners

# number of steps/episodes, we kept it contained for computational reasons. 
N_EPISODES = 8000
DQN_STEPS = 25000


def main():
    # the optimum, on a dense grid 
    print("optimum (stochastic descent) ...", flush=True)
    F_opt = np.array([stochastic_descent(T, n_steps=N_STEPS, n_restarts=30)[0] for T in T_DENSE])
    print("agents ...", flush=True)
    # the learners, at a few T values 
    F_simple, F_bukov, F_dqn, F_smart = [], [], [], []
    for T in T_AGENTS:
        # opt reference 
        F_comparison = stochastic_descent(T, n_steps=N_STEPS, n_restarts=30)[0]
        # simple tabular Q
        _, _, fs, _ = train_simple(T=T, n_steps=N_STEPS, n_episodes=N_EPISODES)

        # Bukov 
        _, _, fb, _, _ = train_bukov(T=T, n_steps=N_STEPS, n_episodes=N_EPISODES)

        # vanilla DQN: one-step bootstrapping, epsilon-greedy
        _, _, fd, _ = train_dqn(T=T, n_steps=N_STEPS, total_timesteps=DQN_STEPS)

        # smart DQN
        _, _, _, _, fsm_best, _ = train_dqn_smart(
            T=T, n_steps=N_STEPS, n_episodes=N_EPISODES)

        F_simple.append(fs)
        F_bukov.append(fb)
        F_dqn.append(fd)
        F_smart.append(fsm_best)
        print(f"T={T:>4} Opt={F_comparison:.3f} simpleQ={fs:.3f} Bukov={fb:.3f}  "
              f"DQN={fd:.3f} DQN-smart={fsm_best:.3f}", flush=True)

    #  the plot
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(T_DENSE, F_opt, "-", color="gray", lw=2, label="optimum (stochastic descent)")
    ax.plot(T_AGENTS, F_bukov, "o", color="tab:red", ms=9, label=r"Bukov Q($\lambda$)")
    ax.plot(T_AGENTS, F_simple, "s", color="tab:orange", ms=8, label="simple tabular Q")
    ax.plot(T_AGENTS, F_dqn, "^", color="tab:blue", ms=8, label="DQN vanilla (1-step, eps-greedy)")
    ax.plot(T_AGENTS, F_smart, "D", color="tab:green", ms=8,label="DQN smart (best found, still blind)")

    # mark the two phase boundaries and shade the three phases
    ax.axvline(T_C, ls="--", color="gray", lw=1)
    ax.axvline(T_QSL, ls="--", color="gray", lw=1)
    ax.text(T_C, 1.04, r"$T_c$", ha="center")
    ax.text(T_QSL, 1.04, r"$T_{QSL}$", ha="center")
    ax.fill_betweenx([0, 1.1], 0, T_C, alpha=0.06, color="red")
    ax.fill_betweenx([0, 1.1], T_C, T_QSL, alpha=0.06, color="orange")
    ax.fill_betweenx([0, 1.1], T_QSL, 4.0, alpha=0.06, color="green")
    ax.text(T_C / 2, 0.03, "I", ha="center")
    ax.text((T_C + T_QSL) / 2, 0.03, "II", ha="center")
    ax.text((T_QSL + 4) / 2, 0.03, "III", ha="center")

    ax.set_xlabel("protocol duration  T")
    ax.set_ylabel(r"final fidelity  $F_h(T)$")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Ways to prepare the qubit (single qubit, $N_T$ = {N_STEPS})")

    fig.tight_layout()
    fig.savefig("comparison_qubit.png", dpi=130)
    print("DONE -> comparison_qubit.png", flush=True)


if __name__ == "__main__":
    main()
