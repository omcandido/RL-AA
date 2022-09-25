from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
import random

from argumentation.classes import ArgumentationFramework, ValuebasedArgumentationFramework
class Agent(ABC):
    """Abstract Agent class for RL agents
    """
    def __init__(self, alpha: float, gamma: float, epsilon: float):
        """Initialize Agent class.

        Args:
            alpha (float): learning rate
            gamma (float): discount rate
            epsilon (float): exploration rate
        """
        # set up the value of epsilon
        self.alpha = alpha  # learning rate or step size.
        self.gamma = gamma  # discount factor.
        self.epsilon = epsilon  # exploration rate.

    # Choose action at state based on epsilon-greedy policy and valueFunction
    def select_action(self, state, is_greedy: bool = True, allowed_actions:list = None) -> int:
        """Select an action for the given state.

        Args:
            state (_type_): current state.
            is_greedy (bool, optional): if True, the agent does not explore (greedy policy). Otherwise, it explores with probability epsilon. Defaults to True.
            allowed_actions (list, optional): a mask of actions that the agent can choose from. If None, no mask is applied. Defaults to None.

        Returns:
            int: index of the chosen action.
        """
        # Placeholder variable for the chosen action.
        best_actions = None
        # Generate a random number in the range [0,1).
        rand = np.random.rand()
        # Estimated values at current step.
        values = self.values(state)
        # Check if we have to explore.
        if rand < self.epsilon and is_greedy == False:
            # (Wrap it in a list to make it compatible with np.random.choice() in the return statement)
            if allowed_actions != None:
                best_actions = allowed_actions
            else:
                best_actions = [np.random.randint(len(values))]
        else: # Otherwise, we exploit.
            # Get the argmax(values)
            # (we cannot use np.argmax directly, because we cannot randomly break ties).
            if allowed_actions != None:
                best_actions = np.argwhere(values == np.amax(values[allowed_actions])).flatten().tolist()
                best_actions = set(best_actions).intersection(set(allowed_actions))
                best_actions = list(best_actions)
            else:
                best_actions = np.argwhere(values == np.amax(values)).flatten().tolist()

        chosen = int(np.random.choice(best_actions))
        return chosen

    @abstractmethod
    def value(self, state, action) -> float:
        """Gets the value of performing an action at a given state.

        Args:
            state (_type_): current state
            action (_type_): current action

        Returns:
            float: q_hat(state, action)
        """
        pass

    # Return vector of estimated action values of given state, for each action
    @abstractmethod
    def values(self, state) -> np.ndarray:
        """Gets the values of all possible actions at a given state.

        Args:
            state (_type_): current state

        Returns:
            np.ndarray: v_hat(state)
        """
        pass

    # learn with given state, action and target
    @abstractmethod
    def learn(self, state, action, next_state, reward: float, done: bool = False) -> int:
        """Update the parameters of the agent according to the received reward.

        Args:
            state (_type_): current state.
            action (_type_): current action
            next_state (_type_): next_state
            reward (float): current reward
            done (bool, optional): True if next_state is terminal, false otherwise. Defaults to False.

        Returns:
            int:  index of the next action.
        """
        pass

    # Return estimated state value, based on the estimated action values
    def state_value(self, state) -> float:
        """Return the value of the greedy action at the current state.

        Args:
            state (_type_): current state

        Returns:
            float: max(v_hat(state))
        """
        return np.max(self.values(state))


class AAAgent(ABC):
    """Abstract class for the Abstract Argumentation Agent. This agent uses a VAF as its inference engine.
    """
    def __init__(self, vaf:ValuebasedArgumentationFramework, args_actions:dict, observation_to_premises:callable, premises_to_arguments:callable):
        """Initialise AAAgent.

        Args:
            vaf (ValuebasedArgumentationFramework): Value-based argumentation framework.
            args_actions (dict): dictionary of arguments with the index of their corresponding action.
            observation_to_premises (callable): function that transforms the observations of the game to premises.
            premises_to_arguments (callable): function that returns a list of valid arguments given a list of premises.
        """
        self.vaf = vaf
        self.args_actions = args_actions
        self.observation_to_premises = observation_to_premises
        self.premises_to_arguments = premises_to_arguments
        self.reset_memory()

    def select_action(self, obs) ->int:
        """Select an action according to the VAF it has been initialised with.

        Args:
            obs (_type_): observation of the game.

        Returns:
            int: index of the selected action.
        """
        vsaf = self.get_vsaf(obs)
        ext = self.get_extension(vsaf)
        action = self.get_extension_action(ext)
        self.update_memory(obs, action)
        return action
    
    @abstractmethod
    def reset_memory(self):
        """Forget about the actions performed in given the previous observations.
        """
        self.memory = []

    @abstractmethod
    def update_memory(self, obs, action):
        """Update the record of actions performed given the current observation of the game.

        Args:
            obs (_type_): current observation of the game.
            action (_type_): current action.
        """
        pass

    def get_vsaf(self, obs) -> ValuebasedArgumentationFramework:
        """Get the value-based situation-specific argumentation framework (VSAF) given the current observation of the game.

        Args:
            obs (_type_): current observation of the game.

        Returns:
            ValuebasedArgumentationFramework: the VSAF.
        """
        vsaf = deepcopy(self.vaf)
        prems = self.observation_to_premises(obs, self.memory)
        valid_args = self.premises_to_arguments(prems)
        invalid_args = set(vsaf.args) - set(valid_args)
        vsaf.remove_arguments(invalid_args)
        return vsaf

    @staticmethod
    def get_extension(vsaf: ValuebasedArgumentationFramework):
        """Returns the grounded extension. In a total strict order (such as as ours), the grounded extension contains always one argument.

        Args:
            vsaf (ValuebasedArgumentationFramework): the VSAF given the current observation of the game.

        Returns:
            _type_: arguments in the grounded extension.
        """
        sum_cols = vsaf.mat.sum(axis=0)
        indices = np.nonzero(sum_cols==0)
        ext = np.take(vsaf.args, indices)[0]
        return ext
        
    def get_extension_action(self, ext: list) -> int:
        """Gets the action promoted by the arguments in the grounded extension. 

        Args:
            ext (list): the grounded extension.

        Returns:
            int: index of the promoted action.
        """
        if len(ext) == 0:
            # print("No extension: performing random action...")
            return random.sample(sorted(self.args_actions.values()), 1)[0]
        return self.args_actions[ext[0]]