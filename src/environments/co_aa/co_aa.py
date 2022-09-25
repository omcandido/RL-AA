from copy import deepcopy
import gym
import numpy as np
from typing import Callable, Optional, List
import time
import random

from argumentation.classes import ArgumentationFramework, ValuebasedArgumentationFramework
from argumentation.utils import order_to_matrix

from agents.agent import AAAgent

class COAAenv(gym.Env):
    """Combinatorial-Optimisation Abstract-Argumentation environment.
    It creates an environment to solve the problem of ordering the arguments in the AF.
    """

    metadata = {"render_modes": ["ansi"]}
    
    def __init__(
        self,
        args: List[int],
        actions: dict, 
        af: ArgumentationFramework, 
        env: gym.Env, 
        observation_to_premises: Callable,
        premises_to_args: Callable,
        aa_agent: AAAgent
    ):
        """Initialise COAAenv

        Args:
            args (List[int]): list of arguments to order
            actions (dict): action promoted by each argument
            af (ArgumentationFramework): AF input by the domain expert
            env (gym.Env): the game to be played by the VAF
            observation_to_premises (Callable): function that transforms a game observation into a list of premises
            premises_to_args (Callable): function that transforms a list of premises into a list of valid arguments
            aa_agent (AAAgent): the agent that will use the VAF as its inference engine
        """
    
        self._args = args
        self._actions = actions 
        self._af = af
        self._env = env
        self._observation_to_premises = observation_to_premises
        self._premises_to_args = premises_to_args
        self._aa_agent = aa_agent
        self._size = len(args)
        self._order = []
        self._order_idx = []

        self.observation_space = gym.spaces.Box(-1, self._size-1, (1,self._size), 'int')

        self.action_space = gym.spaces.Discrete(self._size)

    def step(self, action: int):
        """Given the index of an argument, append it to the partial solution and let the environment evolve.

        Args:
            action (int): the index of the appended argument

        Returns:
            _type_: see Gym documentation
        """
        
        if action in self._order_idx:
            reward = -0.01
            done = True
        else:
            self._order_idx.append(action)
            self._order.append(self._args[action])
            done = (len(self._order) == self._size)

            if done:
                vaf = ValuebasedArgumentationFramework(self._af.args, self._af.atts, self._order)
                self._aa_agent.vaf = vaf
                reward = self._get_game_reward()
            else:
                reward = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, done, info

    def _get_game_reward(self, render=False, lapse=0.25):
        """Uses the AAAgent to play the game and returns the reward.

        Args:
            render (bool, optional): whether to render this run or not. Defaults to False.
            lapse (float, optional): if render is True, this will be the time between frames. Defaults to 0.25.

        Returns:
            _type_: the reward output by the game
        """
        current_state = self._env.reset()
        total_reward = 0

        done = False
        while not done:
            current_action = self._aa_agent.select_action(current_state)
            if render:
                self._env.render()
                time.sleep(lapse)
            current_state, reward, done, _ = self._env.step(current_action)
            total_reward += reward

        self._aa_agent.reset_memory()
        return total_reward

    def _get_obs(self):
        """ The observation of this environment is the encoded (partial) ordering of arguments."""
        return order_to_matrix(self._order, self._args, True)

    def _get_info(self):
        return {'order' : self._order}

    def update_agent_vaf(self, vaf):
        """Updates the VAF of the AAAgent to use in the game when the final reward is computed."""
        self._aa_agent.vaf = vaf

    def render(self, mode="ansi"):
        assert mode is None or mode in self.metadata["render_modes"]
        print(self._order)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._order = []
        self._order_idx = []
        self._aa_agent.reset_memory()

        self._env.reset()

        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), self._get_info()