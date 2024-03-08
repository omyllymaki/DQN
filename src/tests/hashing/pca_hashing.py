import unittest

import torch
from matplotlib import pyplot as plt

from src.dqn.state_hashing import PCAStateHashing

torch.manual_seed(42)


class PCAHashingTests(unittest.TestCase):

    def test_return_type(self):
        hashing = PCAStateHashing()
        x = torch.randn(100, 2)
        hashes = hashing.fit(x)
        self.assertTrue(isinstance(hashes, list))
        self.assertTrue(isinstance(hashes[0], tuple))

        x = torch.randn(1, 2)
        hashes = hashing.hash(x)
        self.assertTrue(isinstance(hashes, list))

    def test_that_similar_items_have_same_hash(self):
        hashing = PCAStateHashing()

        x = torch.Tensor([
            [1, 2],  # group 0
            [1.1, 2.2],  # group 0
            [4, 5],  # group 1
            [4.2, 5.1],  # group 1
            [10, 20]  # group 2
        ])
        hashes = hashing.fit(x)
        self.assertEqual(len(set(hashes)), 3)
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[2], hashes[3])
        self.assertNotEqual(hashes[0], hashes[2])
        self.assertNotEqual(hashes[0], hashes[4])
        self.assertNotEqual(hashes[2], hashes[4])

    def test_dimension_reduction(self):
        n_samples = 10000
        hashing = PCAStateHashing()
        x = torch.randn(n_samples, 10)
        hashes = hashing.fit(x)
        n_uniques_hashes = len(set(hashes))
        print(n_uniques_hashes)

        self.assertLess(n_uniques_hashes, 100)

    def test_fit_and_apply_should_provide_same_hashes_within_same_input_data(self):
        n_samples = 100
        n_dim = 2
        hashing = PCAStateHashing()
        x = torch.randn(n_samples, n_dim) + torch.tensor([10, 5])
        hashes1 = hashing.fit(x)
        hashes2 = hashing.fit(x)

        self.assertEqual(len(hashes1), len(hashes2))
        for (h1, h2) in zip(hashes1, hashes2):
            self.assertEqual(h1, h2)

    def test_grouping_visual_test(self):
        n_samples = 100
        n_dim = 2
        hashing = PCAStateHashing()
        x = torch.randn(n_samples, n_dim) + torch.tensor([10, 5])
        hashes = hashing.fit(x)

        for h in set(hashes):
            i = [hi == h for hi in hashes]
            plt.plot(x[i, 0], x[i, 1], ".")
        plt.show()
