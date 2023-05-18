import random
import numpy as np
import argparse
from gym import Env
from typing import Sequence

import math
from scipy.spatial.distance import cityblock

from agent import Agent
from utils import compare_results

class RandomAgent(Agent):

    def __init__(self):
        super(RandomAgent, self).__init__("Random Agent")
        self.n_actions = 4

    def action(self) -> int:
        return np.random.choice(np.arange(self.n_actions))
    
    def move_direction(self):
        action = self.action()
        if (action == 0):
                return [0,1]
        elif (action == 1):
                return [0,-1]
        elif (action == 2):
                return [1,0]
        elif (action == 3):
                return [-1,0]
'''
class GreedyAgent(Agent):

    def __init__(self, agent_id):
        super(GreedyAgent, self).__init__(f"Greedy Agent")
        self.agent_id = agent_id
        self.n_agents = 2
        self.n_actions = 4

    def action(self) -> int:
        agents_positions = self.observation[:self.n_agents * 2]
        agent_pos = agents_positions[self.agent_id*2:self.agent_id*2+1]
        preys_positions = self.observation[self.n_agents*2 :]
        

        prey_pos = self.closest_prey(agent_pos, preys_positions)
        return self.direction_to_go(agent_pos, prey_pos)
    

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def direction_to_go(self, agent_position, prey_position):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """
        distances = np.array(prey_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)
        if abs_distances[0] > abs_distances[1]:
            return self._close_horizontally(distances)
        elif abs_distances[0] < abs_distances[1]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    def closest_prey(self, agent_position, prey_positions):
        """
        Given the positions of an agent and a sequence of positions of all prey,
        returns the positions of the closest prey.
        If there are no preys, None is returned instead
        """
        min = math.inf
        closest_prey_position = None
        n_preys = int(len(prey_positions) / 2)
        for p in range(n_preys):
            prey_position = prey_positions[p * 2], prey_positions[(p * 2) + 1]
            distance = cityblock(agent_position, prey_position)
            if distance < min:
                min = distance
                closest_prey_position = prey_position
        return closest_prey_position

    # ############### #
    # Private Methods #
    # ############### #

    def _close_horizontally(self, distances):
        if distances[0] > 0:
            return RIGHT
        elif distances[0] < 0:
            return LEFT

    def _close_vertically(self, distances):
        if distances[1] > 0:
            return UP
        elif distances[1] < 0:
            return DOWN
'''