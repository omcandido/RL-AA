import numpy as np
from typing import List, Tuple, MutableSet

def order_to_matrix(
        order: List[str], 
        args: List[str],
        as_bool: bool = False
    ):
    
        """Creates a matrix that encodes the (partial) ordering of arguments.

    Args:
        order (List[str]): current (partial) ordering
        args (List[str]): full list of arguments. The returned matrix preserves indices with this list.
        as_bool (bool, optional): whether the returned matrix should be a Boolean matrix. Defaults to False.

    Returns:
        _type_: returns a matrix with the encoded (partial) ordering
    """

        mat = np.ones((len(args), len(args)))
        indices_args = np.array([args.index(ord) for ord in order if ord !=-1], 'int')
        for i in range(len(indices_args)):
            column = np.zeros(len(args))
            column[indices_args[:i+1]] = 1
            mat[:,indices_args[i]] = column


        mat = np.transpose(mat) # to be consistend with the formulation in the thesis

        if as_bool:
            return mat.astype(bool)
        return mat

def construct_all_attacks(arg_actions: dict) -> MutableSet[Tuple[str, str]]:
    """Given a dictionary of arguments and their promoted action, returns a set with all attacks among them.

    Args:
        arg_actions (dict): dictionary in the format {argument: action}

    Returns:
        MutableSet[Tuple[str, str]]: set of attacks among
    """
    attacks = set()
    for arg1 in arg_actions:
        for arg2 in arg_actions:
            if arg_actions[arg1] != arg_actions[arg2]:
                attacks.add((arg1, arg2))
                attacks.add((arg2, arg1))      
    return attacks