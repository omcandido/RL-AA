import numpy as np
from typing import Tuple
import gym
import time
from agents.agent import Agent
from math import factorial

from environments.frozen_lake.frozen_lake import FrozenLakeWrapper, FrozenLakeNeighboursObservationWrapper, FrozenLakeRewardWrapper
from gym.envs.toy_text.frozen_lake  import generate_random_map

def arrangements(m):
    def variation(m, n):
        return int(factorial(m)/factorial(m-n))
    sum = 0
    for n in range(m+1):
        sum += variation(m, n)
    return sum

def update_average(old, new, i) -> np.array:
    return old + (np.array(new)-old)/i

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='valid')

def run_episode(env: gym.Env,
                agent: Agent,
                initial_state: gym.Space,
                is_learning: bool = True,
                is_animating: bool = False, 
                is_rendering: bool = False) -> Tuple[gym.Space, float]:
    # Initialize reward for episode
    total_reward = 0.0
    # Initialize
    is_greedy = not is_learning
    # Get initial action
    current_state = initial_state
    current_action = agent.select_action(initial_state, is_greedy=is_greedy)

    # Track the rendering
    animation_data = []
    
    # Initialize variables
    next_state = None
    done = False

    while not done:
        
        if is_rendering:
            env.render()
            time.sleep(0.25)
        
        next_state, reward, done, _ = env.step(current_action)
        total_reward += reward

        if is_animating:
            animation_data.append((current_state, next_state, env.t, current_action, total_reward))
            
        # Execute the learning and update the state and action
        # ===================================== #
        # Update q only if the agent is learning.
        if is_learning:
            next_action = agent.learn(current_state, current_action, next_state, reward, done)
        else:
            # When not learning, we exploit.
            next_action = agent.select_action(next_state, is_greedy=True)
        # Save next_state and action for the next step.
        current_state = next_state
        current_action = next_action
        # ===================================== #

    if is_rendering:
        env.render()
        time.sleep(0.25)
        # env.close()
    if is_animating:
            animation_data.append((current_state, None, env.t, None, 0))

    return current_state, total_reward, animation_data


def new_fl_env(map_size=8, p=0.8, multiple_visits=True):
    env = gym.make("FrozenLake-v1",  is_slippery=False, desc=generate_random_map(map_size, p))
    env = FrozenLakeWrapper(env, multiple_visits=multiple_visits)
    env = FrozenLakeNeighboursObservationWrapper(env)
    env = FrozenLakeRewardWrapper(env)
    return env