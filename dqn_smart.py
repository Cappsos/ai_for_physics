"""
DQN version with the same mechanisms used by Bukov et al. without violating the paper setting in which 
the agent only sees the blind observation [t/N_T, h/h_max]. No physical insights are insereted.

Mainly we added the eligibility trace in form of Monte-Carlo returns, so the network gets credit assignment 
even with the sparse reward, and we add a best-episode replay buffer to make sure it doesn't forget the good
protocol once it finds it.

We also use Boltzmann (softmax) exploration with a temperature that warms up over
training, like the tabular agent, instead of epsilon-greedy.

The issue is however that even with these additions the model still has to deal with the fact that  
many different quantum trajectories share the same (t, h)), so the Monte-Carlo target it learns
is actually the average final fidelity over all trajectories passing through that (t, h).

"""

import numpy as np
import torch
import torch.nn as nn

from qubit_env import QubitEnv   # the same blind gym env the vanilla DQN uses


def make_qnet(hidden=32):
    """
    A small MLP: 2 inputs (t/N_T, h/h_max) -> Q-value for each of the 2 actions.
    We keep it small on purpose; the input is only 2-D, a big net would just overfit
    noise.
    
    """
    return nn.Sequential(
        nn.Linear(2, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 2),
    )


def train_dqn_smart(T=2.4, n_steps=28, n_episodes=6000,beta_start=1.0, beta_growth=12.0, lr=1e-3, hidden=32, batch_size=128,
                    buffer_size=50000, updates_per_ep=4, best_fraction=0.25, seed=0):
    """
    Train the smart DQN and return (qnet, fields, greedy_fid, history, best_fid)

    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = QubitEnv(T=T, n_steps=n_steps)
    qnet = make_qnet(hidden)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=lr) #set adam optimizer

    # replay buffer of (obs, action, return)
    buffer = []
    best_transitions = [] # the best episode's (obs, action, return) tuples
    best_fid = 0.0
    best_fields = None
    history = np.zeros(n_episodes)

    def q_of(obs):
        # Q-values for a single observation, no gradient tracking
        with torch.no_grad():
            return qnet(torch.as_tensor(obs, dtype=torch.float32)).numpy()

    for ep in range(n_episodes):
        # temperature warms up over training
        beta = beta_start + beta_growth * (ep / n_episodes)

        obs, _ = env.reset()
        episode = []  # (obs, action) pairs we visited this episode
        done = False
        final_fid = 0.0
        while not done:
            q = q_of(obs)
            # Boltzmann exploration over the blind observation
            p = np.exp(beta * (q - q.max()))
            p /= p.sum()
            a = int(rng.choice(2, p=p))

            episode.append((obs.copy(), a))
            obs, r, done, truncated, info = env.step(a)
            final_fid = r # zero until the last step, then the final fidelity

        history[ep] = final_fid

        # Monte-Carlo target: with gamma = 1 and reward only at the end, the return for every step is simply the final fidelity
        for (o, a) in episode:
            buffer.append((o, a, final_fid))

        # remember the best protocol we've seen (we only use its reward to decide)
        if final_fid > best_fid:
            best_fid = final_fid
            best_transitions = [(o, a, final_fid) for (o, a) in episode]
            best_fields = list(env.protocol)

        # trim the buffer
        if len(buffer) > buffer_size:
            buffer = buffer[-buffer_size:]

        # training a few minibatches of plain Monte-Carlo regression 
        if len(buffer) >= batch_size:
            for _ in range(updates_per_ep):
                idx = rng.integers(len(buffer), size=batch_size)
                batch = [buffer[i] for i in idx]

                # mix in some transitions from the best episode so we keep reinforcing the good protocol (the soft "force-learn")
                if best_transitions:
                    n_best = max(1, int(best_fraction * batch_size))
                    bidx = rng.integers(len(best_transitions), size=n_best)
                    batch += [best_transitions[i] for i in bidx]

                obs_b = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                act_b = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
                ret_b = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)

                # Q(obs, a) for the actions we actually took, regressed onto the return
                q_pred = qnet(obs_b).gather(1, act_b[:, None]).squeeze(1)
                loss = ((q_pred - ret_b) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    greedy_fid, fields = greedy_rollout(env, qnet)
    return qnet, fields, greedy_fid, history, best_fid, best_fields


def greedy_rollout(env, qnet):
    """
    Follow argmax(Q) on the blind observation from start to finish.
    
    """
    obs, _ = env.reset()
    done = False
    final_fid = 0.0
    while not done:
        with torch.no_grad():
            q = qnet(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        a = int(np.argmax(q))
        obs, r, done, truncated, info = env.step(a)
        final_fid = r
    return final_fid, list(env.protocol)


if __name__ == "__main__":
    # simple test run 
    for T, n in [(1.0, 20), (2.4, 24), (3.0, 30)]:
        _, fields, gfid, hist, best, best_fields = train_dqn_smart(
            T=T, n_steps=n, n_episodes=4000, seed=0)
        print(f"T={T:>4}  greedy={gfid:.4f}  best-seen={best:.4f}")
