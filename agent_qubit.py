"""
We define the learning agent.

We start from the plain Q-learning and then reimplement the variation of Bukov et al. 
that make the physics-blind (t, h) observation actually work.

The main additions are:
  1. eligibility traces (Watkins Q(lambda))
        since the reward is sparese (its provided only at the end of the episode), standard one step
        Q learning have an hard time propagint the signal back to the early puleses. Traces allows to carry it back
        in a single sweep. 
  2. a force-learn step (learn_policy)
    it stores the best protocol we've found so far into the Q-table, so we don't drift away from it.

by setting lam = 0 we get the standard Q-learning. We will use it as comparison to Bukov et al. solution alongside the DQN
"""

import numpy as np


class Agent:
    def __init__(self, n_states, n_actions=2,
                 learning_rate=0.4, discount_factor=1.0, lam=0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.df = discount_factor   # gamma is set at 1 since its undiscounted
        self.lam = lam              # lamnda parameter for eligibility traces; lam=0 is standard Q-learning, lam=1 is full Monte Carlo
        self.Q_table = np.zeros((n_states, n_actions))

    

    def greedy_action(self, state):
        # just take the action with the highest Q-value at this state. if there's a tie, take the first one.
        return int(np.argmax(self.Q_table[state]))

    def eps_greedy_action(self, state, eps, rng):
        # with probability eps we explore at random, otherwise we go greedy
        if rng.random() < eps:
            return int(rng.integers(self.n_actions))
        return self.greedy_action(state)

    def softmax_action(self, state, beta, rng):
        # pick an action with probability distribution of exp(beta * Q). small beta is almost
        # random, large beta is almost greedy. we subtract the max first so the  exponential doesn't overflow
        q = self.Q_table[state] 
        p = np.exp(beta * (q - q.max()))                
        p = p / p.sum()                                   
        return int(rng.choice(self.n_actions, p=p))     


    def Qtrain(self, batch, replaying=False):
        """
        Algorithm to run one Watkins Q(lambda) sweep over an episode's transitions.

        batch is the list of (s, a, r, s_next, done) tuples in the order they happened.
        We take a snapshot of Q at the start so we can tell which action was the greedy one at each step,
        since this is what decides when Watkins' rule cuts the trace.
        """
        Q_snapshot = self.Q_table.copy() # copy the Q-table to apply watkins rule correctly
        e = np.zeros_like(self.Q_table)   # eligibility traces, reinitialize at the start of each episode

        for (s, a, r, s_next, done) in batch:
            # The greedy action according to the snapshot of Q at the start of the episode. 
            greedy_a = int(np.argmax(Q_snapshot[s]))

            # one-step TD target; no bootstrap on the terminal step
            if done:
                target = r
            else:
                target = r + self.df * self.Q_table[s_next].max()
            td_error = target - self.Q_table[s, a]

            # replacing trace on the visited (state, action), then nudge all of Q
            e[s, a] = 1.0
            self.Q_table += self.lr * td_error * e  # update Q-table with the TD error weighted by the eligibility traces

            # Watkins' rule: if we took the greedy action (or we're deliberately replaying the best protocol) 
            # the trace chain is still valid, so we let it live and decay it by gamma*lambda. 
            # Otherwise  we reset the traces
            if replaying or a == greedy_a:
                e *= self.df * self.lam
            else:
                e[:] = 0.0

    def learn_policy(self, best_traj, R):
        """
        Force-learn step (Bukov's Learn_Policy)
        we clean the Q-table so at each step we only have one action with non-zero Q-value,
        the one taken in the best protocol, and we set it to the reward R obtained by that protocol.
        
        "best_traj" is the list of (state, action) pairs along the best episode.
        """
        for s, a in best_traj:
            self.Q_table[s, :] = 0.0
            self.Q_table[s, a] = R

    def __repr__(self):
        return f"Agent(n_states={self.n_states}, n_actions={self.n_actions}, lam={self.lam})"
