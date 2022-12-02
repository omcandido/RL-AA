import gym
from typing import Callable, Optional, List
import time
import subprocess
import socket

from argumentation.classes import ArgumentationFramework, ValuebasedArgumentationFramework
from argumentation.utils import order_to_matrix

from agents.agent import AAAgent

class COAATakeaway(gym.Env):
    """Combinatorial-Optimisation Abstract-Argumentation environment that communicates with RoboCup via sockets.
    It creates an environment to solve the problem of ordering the arguments in the AF.
    """

    metadata = {"render_modes": ["ansi"]}
    
    def __init__(
        self,
        args: List[int],
        send_host: str,
        send_port: int,
        recv_host: str,
        recv_port: int,
        ordering_path: str
    ):
        """Initialise COAATakeaway

        Args:
            args (List[int]): list of arguments to order
            actions (dict): action promoted by each argument
        """
    
        self._args = args
        self._size = len(args)
        self._order = []
        self._order_idx = []
        self._send_host = send_host
        self._send_port = send_port
        self._recv_host = recv_host
        self._recv_port = recv_port
        self._ordering_path = ordering_path

        self.observation_space = gym.spaces.Box(0, self._size-1, (1,self._size), 'int')
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
                self._save_ordering()
                reward = self._get_game_reward()
            else:
                reward = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, done, info

    def _get_game_reward(self):
        """Send a message to RoboCup to start playing and listen until it receives the final reward.

        Returns:
            _type_: the reward output by the game
        """

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto("start".encode(), (self._send_host, self._send_port))
            print("Start sent")


        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((self._recv_host, self._recv_port))
            reward = ""
            while reward == "":
                reward, addr = s.recvfrom(16)
                reward = float(reward.decode("utf-8"))
                reward = - reward
                print("  Reward: %s" % str(reward))

        return reward

    def _save_ordering(self):
        f = open(self._ordering_path, "w")
        for i, elem in enumerate(self._order_idx):
            f.write("{} {}\n".format(elem, self._size-i))
        f.close()    

    def _get_obs(self):
        """ The observation of this environment is the encoded (partial) ordering of arguments."""
        return order_to_matrix(self._order, self._args, True)

    def _get_info(self):
        return {'order' : self._order}

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

        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), self._get_info()