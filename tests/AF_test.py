import unittest
import numpy as np
from src.argumentation.utils import construct_all_attacks, order_to_matrix

class Test(unittest.TestCase):
    def test_construct_all_attacks(self):
        arg_actions = {
            'U': 'UP',
            'L': 'LEFT',
            'R': 'RIGHT',
            'D': 'DOWN',
            'gD': 'DOWN',
            'gR': 'RIGHT'
        }

        atts = {('U', 'L'), ('U','R'), ('U','D'), ('U', 'gD'), ('U','gR'),
                ('L', 'U'), ('L','R'), ('L','D'), ('L', 'gD'), ('L','gR'),
                ('R', 'L'), ('R','U'), ('R','D'), ('R', 'gD'),
                ('D', 'L'), ('D','R'), ('D','U'), ('D', 'gR'),
                ('gR', 'L'), ('gR','U'), ('gR','D'), ('gR', 'gD'),
                ('gD', 'L'), ('gD','U'), ('gD','R'), ('gD', 'gR')}

        constructed_atts = construct_all_attacks(arg_actions)
        
        self.assertEqual(atts, constructed_atts)

    def test_ordering_matrix_enconding(self):
        args = ['a', 'b', 'c', 'd', 'e']
        order  = ['d','a', 'e', 'b', 'c']

        expected_encoding = np.array([
            [1., 0., 0., 1., 0.],
            [1., 1., 0., 1., 1.],
            [1., 1., 1., 1., 1.],
            [0., 0., 0., 1., 0.],
            [1., 0., 0., 1., 1.]
        ])

        encoding = order_to_matrix(order, args)
        
        np.testing.assert_array_equal(expected_encoding, encoding)

if __name__ == '__main__':
    unittest.main()