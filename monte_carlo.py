""" Monte Carlo Agent """
from random import random, seed
from env import Easy21
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

class MonteCarloAgent(object):
    """
        Class that implements a Monte Carlo Control agent for model free control
            - states and action-value functions are tabular (no approximators)
            - state = number 1-10, number 1-21
        The agent works like this:
            - collect some samples from an episode i.e. play the game until completion
              according to current greedy policy
            - for each state and action:
                - update times that state was visited N(S_t)
                - update state-action count N(S_t, A_t)
                - update Q(S_t, A_t) = Q(S_t, A_t) + 1 / N(S_t, A_t) * (G_t - Q(S_t, A_t))
    """
    def __init__(self, EPS_CONST=100):
        self._EPS_GREEDY_CONST = EPS_CONST
        # 10 x 21 matrix where each cell is the number of times a state has been visited
        # 10 x 21 matrix of dicts that associates to each state + action the number of times
        # 10 x 21 matrix of dicts that represent action-value function Q(S,A)
        self._Ns = [[None] * 21] * 10
        self._Nsa = [[None] * 21] * 10
        self._Qsa = [[None] * 21] * 10
        for i in range(10):
            for j in range(21):
                self._Ns[i][j] = 0
                self._Nsa[i][j] = {'stick': 0, 'hit': 0}
                self._Qsa[i][j] = {'stick': 0, 'hit': 0}

        self._easy21 = Easy21()

    def learn(self, episodes=10000):
        """
            Run learning algorithm for a number of episodes
        """
        e = 0
        while e < episodes:
            samples = self._collect_samples()
            for sample in samples:
                cs = sample['current_state']
                ds = cs['dealer_sum'] - 1
                ps = cs['player_sum'] - 1
                a = sample['action']
                G_t = sample['reward']

                self._Qsa[ds][ps][a] = self._Qsa[ds][ps][a] + 1.0 / self._Nsa[ds][ps][a] * (G_t - self._Qsa[ds][ps][a])
            e += 1
            self._easy21.restart()
        
    def _collect_samples(self) -> list:
        """
            Generate samples of the form:
                {s_t, a_t, r_t, s_t+1} where r_t is the cumulative reward at timestep t
        """
        current_state: dict = self._easy21.get_state()  # get start state

        action: str = self._act(current_state)
        transition_result = self._easy21.step(action)
        cumulative_reward = transition_result['reward'] # cumulative reward at each timestep

        samples = [
            {
                'current_state': current_state,
                'action': action,
                'reward': cumulative_reward,
                'successor_state': transition_result['state']
            }
        ]

        # update values for current state and state-action
        ds = current_state['dealer_sum'] - 1
        ps = current_state['player_sum'] - 1
        self._Ns[ds][ps] += 1
        self._Nsa[ds][ps][action] += 1
        
        while not transition_result['terminal']:
            current_state = transition_result['state'] # successor state
            action = self._act(current_state)
            transition_result = self._easy21.step(action)
            cumulative_reward += transition_result['reward']

            samples += [
                {
                    'current_state': current_state,
                    'action': action,
                    'reward': cumulative_reward,
                    'successor_state': transition_result['state']
                }
            ]
            ds = current_state['dealer_sum'] - 1
            ps = current_state['player_sum'] - 1
            self._Ns[ds][ps] += 1
            self._Nsa[ds][ps][action] += 1
        return samples
    

    def _act(self, state: dict) -> str:
        """
            Act eps-greedily wrt to the current policy
        """
        seed()
        epsilon = self._EPS_GREEDY_CONST / (self._EPS_GREEDY_CONST + self._Ns[state['dealer_sum'] - 1][state['player_sum'] - 1])
        if random() < epsilon:
            return self._random_action()
        else:
            return self._best_action(state)

    def _best_action(self, state: dict) -> str:
        ds = state['dealer_sum'] - 1
        ps = state['player_sum'] - 1
        hit_val = self._Qsa[ds][ps]['hit']
        stick_val = self._Qsa[ds][ps]['stick']

        if stick_val > hit_val:
            return 'stick'
        elif stick_val < hit_val:
            return 'hit'
        else:
            return self._random_action()

    def _random_action(self):
        seed()
        if random() < .5:
            return 'hit'
        else:
            return 'stick'

    def plot_value_fun(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Make the X, Y meshgrid.
        xs = np.arange(1, 11)
        ys = np.arange(1, 22)
        X, Y = np.meshgrid(xs, ys)
        zs = np.array([max(self._Qsa[x-1][y-1]['hit'], self._Qsa[x-1][y-1]['stick']) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z)

        ax.set_xlabel('Dealer starting card')
        ax.set_ylabel('Player sum')
        ax.set_zlabel('Value')

        plt.show()

    def get_Q(self) -> list:
        return self._Qsa
