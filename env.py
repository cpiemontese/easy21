"""
Environment for Easy21

Rules:
    - inf cards (sampled with replacement)
    - draw = card from 1 to 10 (uniform) with 1/3 prob of being Red and 2/3 of being Black
    - no aces or pictures
    - at the start of the game both player and dealer draw a black card fully observed
    - each turn the player can stick or hit
        - hit = draw a card
        - sticks = no card for the rest of the game
    - black cards add value, red cards remove value
    - if the sum of the player > 21 or < 1 then he/she goes bust i.e. loses (reward -1)
    - if the player sticks then the dealer starts taking turns:
        - the dealer always sticks on any sum >= 17, hits otherwise
            - if the dealer goes bust then the player wins i.e. reward +1
            - if the dealer doesn't bust then the player with the highest sum wins (draw = 0)
"""
import random
import math

class Easy21(object):
    """
        Easy21 environment:
            - state = player sum, dealer sum
    """
    def __init__(self):
        random.seed()

        # init game by drawing a card for both player and dealer
        self._player_sum: int = self._random_card(True)
        self._dealer_sum: int = self._random_card(True)


    def _random_card(self, force_black: bool = False) -> int:
        """
        Return a random card, black with probability 2/3
        i.e. rand falls into [0,.66) or [.66, 1)
        """
        card = 0
        color_choice = random.random()
        if force_black or color_choice < 0.66:
            card = self._rand_card()
        else:
            card = -self._rand_card()
        return card

    def step(self, action: str) -> dict:
        """
            Given and action (stick or hit) compute
                {s', r, terminal}
        """
        terminal: bool = False
        reward: int = 0

        if action == 'stick':
            # play out the game
            while self._dealer_sum < 17 and self._dealer_sum > 1:
                self._dealer_sum += self._random_card()

            # check if dealer busts but state is terminal nevertheless
            terminal = True
            if self._dealer_sum < 1 or self._dealer_sum > 21:
                reward = 1 # win
            else:
                reward = self._compute_reward() # reward get computed based on sums
        elif action == 'hit':
            self._player_sum += self._random_card()
            # check if player busts, if not the game can go on
            if self._player_sum < 1 or self._player_sum > 21:
                terminal = True
                reward = -1
            # else leave reward and terminal as default i.e. 0 and False

        return {
            'state': self.get_state(),
            'reward': reward,
            'terminal': terminal
        }

    def _compute_reward(self) -> int:
        reward = 0
        if self._dealer_sum > self._player_sum:
            reward = -1
        elif self._dealer_sum < self._player_sum:
            reward = 1
        return reward

    def get_state(self) -> dict:
        """ Get current state """
        return {'player_sum': self._player_sum, 'dealer_sum': self._dealer_sum}

    def restart(self):
        """ Restart game i.e. redraw """
        self._player_sum: int = self._random_card(True)
        self._dealer_sum: int = self._random_card(True)

    def _rand_card(self) -> int:
        """ Return a random Easy21 card i.e. a number from 1 to 10 """
        return math.floor(random.random() * 10 + 1)
