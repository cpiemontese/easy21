""" Monte Carlo Agent """
from random import random, seed
from env import Easy21
from mpl_toolkits.mplot3d import Axes3D

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

class SarsaLambdaAgent(object):
    """
        Class that implements a Sarsa(lambda) Control agent for model free control
            - states and action-value functions are tabular (no approximators)
            - state = number 1-10, number 1-21
        The agent works like this:
            - for each episode:
              - init eligibility traces to 0
              - init S and A
              - for each step:
                - take an action A, observe (R, S')
                - choose A' from S' given a policy wrt Q
                - delta = R + gamma * Q(S', A') - Q(S, A)
                - E(S, A) = E(S, A) + 1
                - for each state and action:
                    - Q(s,a) = Q(s, a) + alpha * delta * E(s,a)
                    - E(s, a) = gamma * lambda * E(s, a)
                S := S', A := A'
    """
    def __init__(self, LAMBDA=0.0, EPS_CONST=100.0):
        self._LAMBDA: float = LAMBDA
        self._EPS_GREEDY_CONST: float = EPS_CONST
        # 10 x 21 matrix where each cell is the number of times a state has been visited
        # 10 x 21 matrix of dicts that represent action-value function Q(S,A)
        # 10 x 21 eligibility traces matrix
        self._Ns = [[None] * 21] * 10
        self._Nsa = [[None] * 21] * 10
        self._Qsa = [[None] * 21] * 10
        self._Esa = [[None] * 21] * 10
        for i in range(10):
            for j in range(21):
                self._Ns[i][j] = 0
                self._Nsa[i][j] = {'stick': 0, 'hit': 0}
                self._Qsa[i][j] = {'stick': 0, 'hit': 0}

        self._easy21 = Easy21()

    def set_lambda(self, new_lambda):
        self._LAMBDA = new_lambda

    def _et_init(self):
        for i in range(10):
            for j in range(21):
                self._Esa[i][j] = {'stick': 0, 'hit': 0}

    def learn(self, episodes=1000):
        """
            Run learning algorithm for a number of episodes
        """
        e = 0
        while e < episodes:
            self._et_init()                             # init eligibility traces to 0
            current_state = self._easy21.get_state()    # init S
            current_action = self._act(current_state)   # init A

            terminal = False

            while not terminal:
                observation = self._easy21.step(current_action)
                R = observation['reward']
                successor_state = observation['state']
                terminal = observation['terminal']

                ssds = successor_state['dealer_sum']-1
                ssps = successor_state['player_sum']-1
                csds = current_state['dealer_sum']-1
                csps = current_state['player_sum']-1

                # update Ns and Nsa variables --> not part of Sarsa but required for alpha and eps-greedy
                self._Ns[csds][csps] += 1
                self._Nsa[csds][csps][current_action] += 1

                 # select an action from the new state provided that it is not a terminal state
                if not terminal:
                    new_action = self._act(successor_state) # compute A' from state S'
                    Q_succ = self._Qsa[ssds][ssps][new_action]
                else:
                    Q_succ = 0.0 # a reward from a terminal state is always 0

                Q_curr = self._Qsa[csds][csps][current_action]
                delta = R + Q_succ - Q_curr
                self._Esa[csds][csps][current_action] += 1  # update current state-action eligibility trace

                for i in range(10):
                    for j in range(21):
                        for a in ['stick', 'hit']:
                            if self._Nsa[i][j][a] > 0:
                                self._Qsa[i][j][a] += 1.0 / self._Nsa[i][j][a] * delta * self._Esa[i][j][a]
                                self._Esa[i][j][a] *= self._LAMBDA * self._Esa[i][j][a]

                if not terminal:
                    current_state = successor_state
                    current_action = new_action
            e += 1
            self._easy21.restart()

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
