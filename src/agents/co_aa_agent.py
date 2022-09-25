import gym

import numpy as np
from agents.agent import Agent
from argumentation.classes import ArgumentationFramework, ValuebasedArgumentationFramework
from argumentation.utils import order_to_matrix
from typing import List

class COAAAgent(Agent):
    """Combinatorial-Optimisation Abstract-Argumentation Agent: the agent that learns the VAF from the input AF.

    Args:
        Agent (_type_): _description_
    """
    def __init__(self, alpha: float, gamma: float, epsilon: float, args: List[str]):
        """Initialisatialise the COAAAgent.

        Args:
            alpha (float): learning rate.
            gamma (float): discount factor
            epsilon (float): exploration rate
            args (List[str]): list of arguments to be ordered.
        """
        super().__init__(alpha, gamma, epsilon)
        self.args = args
        self.W_SHAPE = (len(args), len(args), len(args))
        self.w = np.zeros(self.W_SHAPE)

    # Return estimated action value of given state and action
    def value(self, state, action) -> float:
        """Gets the value of performing an action at a given state.

        Args:
            state (_type_): current state
            action (_type_): current action

        Returns:
            float: q_hat(state, action)
        """
        w_active = self.w[state, action]
        val = np.sum(w_active)
        return val

    # Return vector of estimated action values of given state, for each action
    def values(self, state) -> np.ndarray:
        """Gets the values of all possible actions at a given state.

        Args:
            state (_type_): current state

        Returns:
            np.ndarray: v_hat(state)
        """
        w_active = self.w[state, :]
        vals = np.sum(w_active, axis=0)
        return vals

    # Return estimated state value, based on the estimated action values
    def state_value(self, state) -> float:
        """Return the value of the greedy action at the current state.

        Args:
            state (_type_): current state

        Returns:
            float: max(v_hat(state))
        """
        return np.max(self.values(state))

    # learn with given state, action and target
    def learn(self, state: gym.Space, action, next_state, reward: float, done: bool = False):
        """Update the parameters of the agent according to the received reward using SARSA.

        Args:
            state (_type_): current state.
            action (_type_): current action
            next_state (_type_): next_state
            reward (float): current reward
            done (bool, optional): True if next_state is terminal, false otherwise. Defaults to False.

        Returns:
            int:  index of the next action.
        """


        mask = np.sum(next_state, axis=0) == len(next_state)
        allowed_actions = np.argwhere(mask).flatten().tolist()
        # allowed_actions = None
        next_action = self.select_action(next_state, False, allowed_actions=allowed_actions)

        if done:
            self.w[state, action] += self.alpha * (reward - self.value(state, action))
            return None

        self.w[state, action] += self.alpha * (reward + self.gamma * self.value(next_state, next_action) - self.value(state, action))
        
        
        return next_action

    @property
    def order(self) -> list:
        """Solution decoded by the agent using a greedy search.

        Returns:
            list: the decoded ordered arguments
        """
        order = []
        for i in range(len(self.args)):
            encoded_order = order_to_matrix(order, self.args, True)
            mask = np.sum(encoded_order, axis=0) == len(encoded_order)
            allowed_actions = np.argwhere(mask).flatten().tolist()
            action = self.select_action(encoded_order, is_greedy=True, allowed_actions=allowed_actions)
            order.append(self.args[action])

        return order

    def is_goal_reached(self):
        NotImplemented