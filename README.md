# Reinforcement Learning for Single-Qubit Quantum State Preparation

Exam project for **AI Models for Physics** - Nicolò Cappa (ID 907094).

Reproduction of the single-qubit case of Bukov et al., *Reinforcement Learning in
Different Phases of Quantum Control*, **Phys. Rev. X 8, 031086 (2018)**
(original code: <https://github.com/mgbukov/dynamicQL>).

Full report: **[Report_project_AI_for_physics.pdf](Report_project_AI_for_physics.pdf)**.

## What it does

A blind RL agent learns to drive a qubit from a start state to a target by choosing a
bang-bang field `h ∈ {−4, +4}`, seeing only `(time step, current field)` and a single
reward (the final fidelity at the end). Five methods are compared across the three
control phases: stochastic descent (the optimum), simple tabular Q-learning, Bukov's
Q(λ), a vanilla DQN, and a "smart" blind DQN.

## How to run

install requirements from requirements.txt

```bash
python main_qubit.py      # train the Bukov Q(λ) agent at T=2.4, save logs + plots
python compare_qubit.py   # main figure: all methods vs T  ->  comparison_qubit.png
python benchmark.py       # the stochastic-descent optimum (quick check)
```


## Layout

- `physics.py` : Hamiltonian, time evolution, fidelity
- `environment_qubit.py`, `agent_qubit.py`, `main_qubit.py` : the tabular agents (Q-learning and Bukov Q(λ))
- `benchmark.py` : stochastic descent (the optimum)
- `train_dqn.py`, `dqn_smart.py` : the two DQNs
- `compare_qubit.py`: the figure 
