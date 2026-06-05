"""
This is a readaptation of the laboratory training loop. 

The episode() function plays each episode and trains the agent on it. 
We then run several episodes and log the results to plot the learning curve

We have two training routines, one for the simple Q-learning agent and one for Bukov-style agent. 
"""

import os
import numpy as np
import pandas
import matplotlib.pyplot as plt

from environment_qubit import Environment, N_FIELDS
from agent_qubit import Agent
from file_manager import csv_writer
from moving_average import plot_movingAverage




def run_episode(env, agent, rng, policy="eps", eps=0.0, beta=0.0, best_actions=None):
    """
    Play a single episode and train the agent on it.

    the pareameter "policy" picks how we choose actions each step:
      - "eps" : epsilon-greedy (the simple agent) it requires "eps" as param
      - "softmax": Boltzmann with temperature beta (Bukov, exploring) it needs "beta" as param
      - "replay" : just follow "best_actions" (which should be a list of actions)

    We collect the transitions into a batch and, at the end of the episode, train
    the agent on the whole batch at once
    """
    env.reset()
    batch = []
    t = 0
    while not env.is_ended():
        s = env.state
        if policy == "eps":
            a = agent.eps_greedy_action(s, eps, rng)
        elif policy == "softmax":
            a = agent.softmax_action(s, beta, rng)
        else:  # "replay"
            a = best_actions[t]
        info = env.step(a)          # (s, a, r, s_next, done)
        batch.append(info)
        t += 1

    agent.Qtrain(batch, replaying=(policy == "replay"))

    # the reward on the last transition is the final fidelity of this episode
    final_fidelity = batch[-1][2]
    return final_fidelity, batch


def greedy_rollout(env, agent):
    """Follow the greedy policy from start to finish and report what we get."""
    env.reset()
    fields = []
    while not env.is_ended():
        a = agent.greedy_action(env.state)
        env.step(a)
        fields.append(env.fields[a])
    return env.reward(), fields    # env.reward() is the fidelity once we're done



def train_simple(T=2.4, n_steps=48, n_episodes=15000, seed=0):
    """
     Simple one-step tabular Q-learning (the baseline)
     
     """
    rng = np.random.default_rng(seed)
    env = Environment(T=T, n_steps=n_steps)
    agent = Agent(n_states=env.n_states, lam=0.0)   # lam = 0 -> ordinary Q-learning

    history = np.zeros(n_episodes)
    for ep in range(n_episodes):
        # epsilon decays linearly from 1 down to a small floor
        eps = max(0.02, 1.0 - ep / n_episodes)
        fid, _ = run_episode(env, agent, rng, policy="eps", eps=eps)
        history[ep] = fid

    final_fid, fields = greedy_rollout(env, agent) # check the final fidelity on a full greedy rollout
    return agent, fields, final_fid, history


def train_bukov(T=2.4, n_steps=48, n_episodes=15000, lam=0.9, beta_start=2.0, beta_growth=8.0,
                replay_every=40, seed=0, verbose=False):
    """
    The Bukov-style agent. We perform Q(lambda) with softmax exploration with beta that warms up over
    training and add alternating blocks of exploration and best-protocol replay
    """
    rng = np.random.default_rng(seed)
    env = Environment(T=T, n_steps=n_steps)
    agent = Agent(n_states=env.n_states, lam=lam)

    best_fid = 0.0
    best_traj = None  # list of (state, action) along the best episode
    best_actions = None  # just the action sequence, for the replay blocks
    history = np.zeros(n_episodes)

    for ep in range(n_episodes):
        # The schedule alternates blocks of replay and explore. 
        # It first spend a block exploring, then spend a block replaying the best protocol we know. 
        # This mirrors the explore/replay stages in the paper. We can only replay once we have actually found a best protocol.

        replaying = (best_actions is not None) and ((ep // replay_every) % 2 == 1) #alternating and trigger only if found a best protocol to replay

        # the softmax "temperature" beta grows over training, so the policy starts out exploring more and slowly approach the greedy policy.
        beta = beta_start + beta_growth * (ep / n_episodes)

        if replaying:
            fid, batch = run_episode(env, agent, rng, policy="replay",
                                     best_actions=best_actions) # if replay we just play the best so far
        else:
            fid, batch = run_episode(env, agent, rng, policy="softmax", beta=beta) # normal exploration step
        history[ep] = fid

        # update best and burn it into the policy with a force-learn step
        if fid > best_fid:
            best_fid = fid
            best_traj = [(s, a) for (s, a, r, s_next, done) in batch]
            best_actions = [a for (s, a) in best_traj]
            agent.learn_policy(best_traj, best_fid)
            if verbose:
                print(f"  ep {ep:5d}  beta={beta:5.2f}  new best fidelity = {fid:.4f}")

    # before finishing we perform a last force-learn step to ensure we actually return the best policy found.\
    # Since during training the later update sweeps can slowly erode the best protocol we burned in earlier, so we
    # stamp it back in once at the end. Like in the paper. 
    if best_traj is not None:
        agent.learn_policy(best_traj, best_fid)

    final_fid, fields = greedy_rollout(env, agent)
    return agent, fields, final_fid, history, best_fid


def plot_protocol(T, fields, fid, fname):
    # Plot the learned protocol as a step function, so we can eyeball that it has the expected bang-bang structure.
    dt = T / len(fields)
    t = np.arange(len(fields)) * dt
    # step plot, repeating the last value so the final pulse is drawn fully
    plt.figure(figsize=(6, 4))
    plt.step(np.append(t, T), np.append(fields, fields[-1]), where="post", color="tab:red")
    plt.xlabel("time t")
    plt.ylabel(r"field $h_x(t)$")
    plt.ylim(-5, 5)
    plt.title(f"learned protocol, F = {fid:.3f}")
    plt.grid(linestyle=":")
    try:
        plt.savefig(fname)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(fname))
        plt.savefig(fname)
    plt.close()



def main():
    # parameters
    T = 2.4
    n_steps = 48 # dt = T / n_steps is around 0.05 that is the value used in the paper
    n_episodes = 15000
    lam = 0.9
    seed = 0

    # store results and graphs in this directory
    out_dir = "data_qubit/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    graph_dir = out_dir + "Graphs/"

    #train 
    print(f"Training the Bukov-style Q(lambda) agent at T = {T} ...")
    agent, fields, final_fid, history, best_fid = train_bukov( T=T, n_steps=n_steps, n_episodes=n_episodes, lam=lam, seed=seed, verbose=True)
    print(f"best fidelity seen: {best_fid:.4f}")
    print(f"greedy rollout gives: {final_fid:.4f}")

    # store the results reusing labs code
    #log per-episode fidelity to csv 
    fid_file = out_dir + "fidelity.csv"
    with csv_writer(fid_file, "w") as f:
        f.writerow(["fidelity"])
        for fid in history:
            f.writerow([fid])

    # save the Q-table
    np.save(out_dir + "Q", agent.Q_table)

    # learning curve
    data = pandas.read_csv(fid_file)
    plot_movingAverage(data["fidelity"], "fidelity_curve.png", 500, graph_dir)

    # plot the learned protocol, so we can see how it's structured 
    plot_protocol(T, fields, final_fid, graph_dir + "protocol.png")

    print(f"Saved logs and plots under {out_dir}")


if __name__ == "__main__":
    main()
