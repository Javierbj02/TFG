import unittest
from main import round_numbers

class TestRoundNumbers(unittest.TestCase):
    def test_round_numbers(self):
        
        # FUERA DE LOS LÍMITES
        real_position = [-1500, 1100]
        expected_output = [-1437, 1155]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [1500,  5000]
        expected_output = [1357, 4029]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [-170, 2500]
        expected_output = [-187, 2405]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [0, 2700]
        expected_output = [107, 2779]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        
        # DENTRO DE LOS LÍMITES
        
        real_position = [-1345, 1555]
        expected_output = [-1347, 1555]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [-1436, 1156]
        expected_output = [-1437, 1155]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [-1000, 2001]
        expected_output = [-997, 2000]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [-186, 2406]
        expected_output = [-187, 2405]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [108, 2800]
        expected_output = [107, 2799]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [205, 3005]
        expected_output = [207, 3004]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [205, 3005]
        expected_output = [207, 3004]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [-546, 4028]
        expected_output = [-547, 4029]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [-600, 3117]
        expected_output = [-597, 3114]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [-600, 3118]
        expected_output = [-597, 3119]
        self.assertEqual(round_numbers(real_position), expected_output)
        
        real_position = [-605, 3117]
        expected_output = [-607, 3114]
        self.assertEqual(round_numbers(real_position), expected_output)
        


if __name__ == '__main__':
    unittest.main()