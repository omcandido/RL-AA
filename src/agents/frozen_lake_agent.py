from agents.agent import Agent, AAAgent

import numpy as np
import matplotlib.pyplot as plt
from environments.frozen_lake.utils import FLActions, FLObservation, fl_safe_cells

import gym

class FrozenLakeAgent(Agent):
    """Non symbolic agent to solve the Frozen Lake game

    Args:
        Agent (class): The generic Agent class
    """
    def __init__(self, map_size: int, alpha: float, gamma: float, epsilon: float, full=False):
        """Initialise FrozenLakeAgent

        Args:
            map_size (int): width of the (squared) map
            alpha (float): learning rate
            gamma (float): discount factor
            epsilon (float): exploration rate
            full (bool, optional): if True, it uses the entire game observation, otherwise, just the tile index. Defaults to False.
        """
        super().__init__(alpha, gamma, epsilon)
        self.map_size = map_size
        if full:
            self.W_SHAPE = (24+map_size*map_size, len(FLActions))
        else:
            self.W_SHAPE = (map_size*map_size, len(FLActions))
        self.w = np.zeros(self.W_SHAPE)
        self.e = np.zeros(self.W_SHAPE)

    # Return estimated action value of given state and action
    def value(self, observation: FLObservation, action: FLActions):
        if self.is_goal_reached(observation):
            return 0.0
        return np.sum(self.w[observation[-self.W_SHAPE[0]:], action])

    # Return vector of estimated action values of given state, for each action
    def values(self, observation):
        if self.is_goal_reached(observation):
            return np.zeros(len(FLActions))
        return np.sum(self.w[observation[-self.W_SHAPE[0]:],:], axis=0)


    # learn with given state, action and target
    def learn(self, state: FLObservation, action: FLActions, next_state: FLObservation, reward: float, done: bool = False):

        state = state[-self.W_SHAPE[0]:]
        next_state = next_state[-self.W_SHAPE[0]:]

        next_action = self.select_action(next_state, False)

        if done:
            self.w[state, action] += self.alpha * (reward - self.value(state, action))
            return None
        
        #Either SARSA or Q-learning can be used, although Q-learning seems to converge faster.
        # self.w[state, action] += self.alpha * (reward + self.gamma*self.value(next_state, next_action) - self.value(state, action))
        self.w[state, action] += self.alpha * (reward + self.gamma*self.state_value(next_state) - self.value(state, action))
        return next_action

    def is_goal_reached(self, state: FLObservation):

        if state[-1]:
            return True
        return False

    # Plot the state value estimates
    def plot_state_values(self):
        """Plots a grid with the values of the state-action space

        Returns:
            _type_: the plot
        """
        v = np.zeros(self.W_SHAPE[0:1])
        for x in range(self.map_size*self.map_size):
            v[x] = self.state_value((x))
        plt.imshow(v)
        plt.colorbar()
        return plt.show()

class FLRandomAgent(Agent):
    """A completely random agent.
    """
    def __init__(self):
        """Agent that takes an action at random.
        """
    
    def select_action(self, state, is_greedy: bool = True) -> int:
        """Choose an action at random.
        """
        return np.random.choice(FLActions)
    
    def value(self, state, action) -> float:
        pass

    def values(self, state) -> np.ndarray:
        pass

    def learn(self, state, action, next_state, reward: float, done: bool = False):
        pass

class FLRandomAwareAgent(Agent):
    """Agent that chooses an action at random from the ones that lead to a safe cell.
    """
    def __init__(self):
        """Agent that takes an action at random.
        """
    
    def select_action(self, state, is_greedy: bool = True) -> int:
        actions = []

        if state[0]:
            actions.append(FLActions.LEFT)
        if state[2]:
            actions.append(FLActions.UP)
        if state[4]:
            actions.append(FLActions.RIGHT)
        if state[6]:
            actions.append(FLActions.DOWN)

        return np.random.choice(actions)
    
    def value(self, state, action) -> float:
        pass

    def values(self, state) -> np.ndarray:
        pass

    def learn(self, state, action, next_state, reward: float, done: bool = False):
        pass

class FLHandcraftedAgent(FLRandomAgent):
    """Agent that follows a handcrafted policy.

    Args:
        FLRandomAgent (_type_): _description_
    """
    def __init__(self):
        """Agent that follows a handcrafted policy.
        """
    def select_action(self, state, is_greedy: bool = True) -> int:
        actions = []
        if state[4]:
            actions.append(FLActions.RIGHT)
        if state[6]:
            actions.append(FLActions.DOWN)

        if len(actions) == 0:
            if state[0]:
                actions.append(FLActions.LEFT)
            if state[2]:
                actions.append(FLActions.UP)

        return np.random.choice(actions)

    # Approach the goal in diagonal.
    def approach_goal(map_size, state):
        dy = (map_size - 1) - state[0]
        dx = (map_size - 1) - state[1]
        if dy < dx:
            action = FLActions.RIGHT
        elif dx < dy:
            action = FLActions.DOWN
        else:
            action = np.random.choice((FLActions.DOWN, FLActions.RIGHT))
        return action

class FLAAAgent(AAAgent):
    def __init__(self, vaf, args_actions, obs_to_prems, prems_to_args, map_size):
        self.map_size = map_size
        super().__init__(vaf, args_actions, obs_to_prems, prems_to_args)

    def reset_memory(self):
        # We want an array where for each tile, we can store what actions we took.
        # There are map_size x map_size x #actions bits to store.
        # Table format is chosen because map tiles are identified by tile index.
        self.memory = np.zeros((self.map_size*self.map_size, len(FLActions)), dtype=bool)

    def update_memory(self, obs, action):
        # Make memory[tile_index][action] = True, 
        # to remember what actions were aready taken in the current tile.
        self.memory[obs[24:]==True, action] = True
