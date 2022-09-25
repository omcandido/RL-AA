
from hashlib import new
from typing import Tuple, List
from enum import IntEnum
import gym
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.animation
from copy import deepcopy


# Frozen Lake observation. E.g., (4,6)
FLObservation = Tuple[int,int]

fl_safe_cells = ('S', 'F', 'G')
fl_unsafe_cells = ('H', '0')


# Frozen Lake Actions.
class FLActions(IntEnum):
    LEFT  = 0
    DOWN  = 1
    RIGHT = 2
    UP    = 3

# This is a shorthand to identify the index of the neighbouring tile. I.e.:
# [1][2][3]
# [0]🤖[4]
# [7][6][5]
class Direction(IntEnum):
    LEFT         = 0
    TOP_LEFT     = 1
    TOP          = 2
    TOP_RIGHT    = 2
    RIGHT        = 4
    BOTTOM_RIGHT = 5
    BOTTOM       = 6
    BOTTOM_LEFT  = 7

arg_actions_naive = {
    'U': FLActions.UP,
    'L': FLActions.LEFT,
    'R': FLActions.RIGHT,
    'D': FLActions.DOWN,
    # 'gD': FLActions.DOWN,
    # 'gR': FLActions.RIGHT,
}

arg_actions_advanced = deepcopy(arg_actions_naive)
arg_actions_advanced.update({
    'ul1-U': FLActions.UP,
    'ul1-L': FLActions.LEFT,
    'ur1-U': FLActions.UP,
    'ur1-R': FLActions.RIGHT,
    'dl1-D': FLActions.DOWN,
    'dl1-L': FLActions.LEFT,
    'dr1-D': FLActions.DOWN,
    'dr1-R': FLActions.RIGHT,
})

arg_actions_advanced2 = deepcopy(arg_actions_naive)
arg_actions_advanced2.update({
    'pu1-D': FLActions.DOWN,
    'pu1-L': FLActions.LEFT,
    'pu1-R': FLActions.RIGHT,
    'pd1-U': FLActions.UP,
    'pd1-L': FLActions.LEFT,
    'pd1-R': FLActions.RIGHT,
    'pl1-U': FLActions.UP,
    'pl1-D': FLActions.DOWN,
    'pl1-R': FLActions.RIGHT,
    'pr1-U': FLActions.UP,
    'pr1-D': FLActions.DOWN,
    'pr1-L': FLActions.LEFT,
})

arg_actions_advanced3 = deepcopy(arg_actions_naive)
arg_actions_advanced3.update({
    'nD': FLActions.DOWN,
    'nL': FLActions.LEFT,
    'nR': FLActions.RIGHT,
    'nU': FLActions.UP
})

arg_actions_advanced4 = arg_actions_advanced2 | arg_actions_advanced3

def fl_map_to_str(map):
    return map.astype('U13')

def fl_plot_run(map, stat_from, stat_to, time, act, rew):
    map_size = len(map)
    def init_func():
        for r in range(map_size):
            for c in range(map_size):
                if map[r,c] == "H":
                    background_color = (0,0,1)
                elif map[r,c] == "S":
                    background_color = (0,1,1)
                elif map[r,c] == "G":
                    background_color = (0,1,0)
                else:
                    background_color = (1,1,1)
                char_x = 0.05 + ((1-0.05)/map_size)*c
                char_y = 1 - (0.1+(((1-0.05)/map_size)*r))
                ax.text(char_x, char_y, map[r,c],
                    fontsize=int(((1-0.2)/map_size)*200),
                    backgroundcolor=background_color,
                    family='monospace')
        # return [artist_state, artist_info]

    def animate(i, map, stat_from, stat_to, time, act, rew):

        def state_to_indices(state):
            """Transform the one-hot part of the state into coordinates in the map"""
            if type(state) is str:
                return state
            n = len(map)
            position = state[24:]
            index = np.argmax(position, axis=0)
            return np.unravel_index(index, (n, n))

        map_size = len(map)
        r,c = state_to_indices(stat_from[i])
        char_x = 0.05 + ((1-0.05)/map_size)*c
        char_y = 1 - (0.1+(((1-0.05)/map_size)*r))
        artist_state.set_x(char_x)
        artist_state.set_y(char_y)
        artist_state.set_text(map[r,c])
        action_str = 'None' if act[i] == 'None' else str(FLActions(act[i]))
        artist_info.set_text("\nTime step: {} \nAction: {} \nState: {} -> {} \nCurrent total reward: {:.4f}".format(time[i], action_str, state_to_indices(stat_from[i]), state_to_indices(stat_to[i]), rew[i]))
        # return [artist_state, artist_info]

    map = fl_map_to_str(map)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.axis('off')
    artist_state = ax.text(0.0, 0., "",
        fontsize=int(((1-0.2)/map_size)*200),
        backgroundcolor=(1,0,0),
        family='monospace',
        zorder=99999)
    artist_info = fig.text(0, 0.01,"")        
    ani = matplotlib.animation.FuncAnimation(fig, animate,
        fargs=(map,  stat_from, stat_to, time, act, rew), 
        init_func=init_func,
        # blit= True,
        frames=len(stat_from),
        interval=100)
    plt.close()
    # ani.save('animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'],)
    return HTML(ani.to_jshtml())


def fl_observation_to_premises(observation, memory) -> dict:
    # Extract relevant vectors for convenience.
    safe = observation[0:8]
    holes = observation[0:16]
    margin = observation[16:24]

    # Get the tile index.
    tile_idx = np.where(observation[24:])[0][0]
    # Get the map size.
    map_size = int(np.sqrt(len(observation)-24))

    # Pad memory and convert to a 3D array for convenience.
    memory_pad = deepcopy(memory)
    memory_pad = memory_pad.reshape(map_size, map_size, 4)
    memory_pad = np.pad(memory_pad, pad_width=((1,1), (1,1), (0,0)))
    
    # Get the coordinates of the current tile in the padded memory.
    row, col = np.unravel_index(tile_idx, (map_size, map_size))
    tile_idx = (row+1)*(map_size+2) + col + 1
    row, col = np.unravel_index(tile_idx, (map_size+2, map_size+2))

    safe_up = safe[Direction.TOP]
    safe_down = safe[Direction.BOTTOM]
    safe_left = safe[Direction.LEFT]
    safe_right = safe[Direction.RIGHT]

    visited_up = any(memory_pad[row-1, col])
    visited_down = any(memory_pad[row+1, col])
    visited_left = any(memory_pad[row, col-1])
    visited_right = any(memory_pad[row, col+1])

    res = {
        'safe_up'    : safe_up,
        'safe_down'  : safe_down,
        'safe_left'  : safe_left,
        'safe_right' : safe_right,
        'visited_up'     : visited_up,
        'visited_down'   : visited_down,
        'visited_left'   : visited_left,
        'visited_right'  : visited_right,
    }
    
    return res

def fl_premises_to_args(premises) -> List[str]:
    args = []
    if premises['safe_up']:
        args.append('U')
    if premises['safe_down']:
        args.append('D')
    if premises['safe_left']:
        args.append('L')
    if premises['safe_right']:
        args.append('R')

    if premises['safe_up'] and not premises['visited_up']:
        args.append('nU')
    if premises['safe_down'] and not premises['visited_down']:
        args.append('nD')
    if premises['safe_left'] and not premises['visited_left']:
        args.append('nL')
    if premises['safe_right'] and not premises['visited_right']:
        args.append('nR')

    # if premises['ul1'] == 'H':
    #     if premises['u1'] in fl_safe_cells:
    #         args.append('ul1-U')
    #     if premises['l1'] in fl_safe_cells:
    #         args.append('ul1-L')

    # if premises['ur1'] == 'H':
    #     if premises['u1'] in fl_safe_cells:
    #         args.append('ur1-U')
    #     if premises['r1'] in fl_safe_cells:
    #         args.append('ur1-R')

    # if premises['dl1'] == 'H':
    #     if premises['d1'] in fl_safe_cells:
    #         args.append('dl1-D')
    #     if premises['l1'] in fl_safe_cells:
    #         args.append('dl1-L')

    # if premises['dr1'] == 'H':
    #     if premises['d1'] in fl_safe_cells:
    #         args.append('dr1-D')
    #     if premises['r1'] in fl_safe_cells:
    #         args.append('dr1-R')
            

    # if premises['prev-u']:
    #     if premises['d1'] in fl_safe_cells:
    #         args.append('pu1-D')
    #     if premises['l1'] in fl_safe_cells:
    #         args.append('pu1-L')
    #     if premises['r1'] in fl_safe_cells:
    #         args.append('pu1-R')

    # if premises['prev-d']:
    #     if premises['u1'] in fl_safe_cells:
    #         args.append('pd1-U')
    #     if premises['l1'] in fl_safe_cells:
    #         args.append('pd1-L')
    #     if premises['r1'] in fl_safe_cells:
    #         args.append('pd1-R')

    # if premises['prev-l']:
    #     if premises['u1'] in fl_safe_cells:
    #         args.append('pl1-U')
    #     if premises['d1'] in fl_safe_cells:
    #         args.append('pl1-D')
    #     if premises['r1'] in fl_safe_cells:
    #         args.append('pl1-R')

    # if premises['prev-r']:
    #     if premises['u1'] in fl_safe_cells:
    #         args.append('pr1-U')
    #     if premises['d1'] in fl_safe_cells:
    #         args.append('pr1-D')
    #     if premises['l1'] in fl_safe_cells:
    #         args.append('pl1-L')
    
    return args


    