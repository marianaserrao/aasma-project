import random
import numpy as np

import math
from scipy.spatial.distance import cityblock

from agent import Agent

N_ACTIONS = 4
DOWN, UP, RIGHT, LEFT = range(N_ACTIONS)

class RandomAgent(Agent):

    def __init__(self):
        super(RandomAgent, self).__init__("Random Agent")
        self.n_actions = 4

    def action(self) -> int:
        return np.random.choice(np.arange(self.n_actions))
        

class GreedyAgent(Agent):

    def __init__(self, agent_id, debug):
        super(GreedyAgent, self).__init__(f"Greedy Agent")
        self.agent_id = agent_id
        self.n_agents = 2
        self.n_actions = 4
        self.debug = debug

    def action(self) -> int:
        agents_positions = self.observation[0][0]
        agent_pos = agents_positions[self.agent_id-1]
        food_pos = self.observation[0][1][self.agent_id-1]
        if self.debug:
            print("Agent", self.agent_id, ": ", agent_pos[0])
            print("Food", self.agent_id,": ", food_pos)
        return self.direction_to_go(agent_pos[0], food_pos)


    # ################# #
    # Auxiliary Methods #
    # ################# #

    def direction_to_go(self, agent_position, food_position):
        """
        Given the position of the agent and the position of a food,
        returns the action to take in order to close the distance
        """
        distances = np.array(food_position) - np.array(agent_position)
        roll = random.uniform(0, 1)
        return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    # ############### #
    # Private Methods #
    # ############### #

    def _close_horizontally(self, distances):
        if distances[0] == 0:
            return self._close_vertically(distances)
        elif distances[0] > 0:
            if self.debug:
                print("Direction: Right")
            return RIGHT
        elif distances[0] < 0:
            if self.debug:
                print("Direction: Left")
            return LEFT

    def _close_vertically(self, distances):
        if distances[1] == 0:
            return self._close_horizontally(distances)
        elif distances[1] > 0:
            if self.debug:
                print("Direction: UP")
            return UP
        elif distances[1] < 0:
            if self.debug:
                print("Direction: Down")
            return DOWN
