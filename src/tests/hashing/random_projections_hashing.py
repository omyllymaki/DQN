import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.dqn.hashing import RandomProjectionHashing

torch.manual_seed(42)


class RandomProjectionHashingTests(unittest.TestCase):

    def test_wrong_data_dim_should_throw_error(self):
        hashing = RandomProjectionHashing(3, 2, scale_factors=5, device="cpu")
        x = torch.randn(100, 3)
        with self.assertRaises(RuntimeError):
            hashing.apply(x)

    def test_return_type(self):
        hashing = RandomProjectionHashing(3, 2, scale_factors=5, device="cpu")
        x = torch.randn(100, 2)
        hashes = hashing.apply(x)
        self.assertTrue(isinstance(hashes, list))
        self.assertTrue(isinstance(hashes[0], tuple))

        x = torch.randn(1, 2)
        hashes = hashing.apply(x)
        self.assertTrue(isinstance(hashes, tuple))

        x = torch.randn(1, 2).view(-1)
        hashes = hashing.apply(x)
        self.assertTrue(isinstance(hashes, tuple))

    def test_that_similar_items_have_same_hash(self):
        hashing = RandomProjectionHashing(3, 2, scale_factors=5, device="cpu")

        x = torch.Tensor([
            [1, 2],  # group 0
            [1.1, 2.2],  # group 0
            [4, 5],  # group 1
            [4.2, 5.1],  # group 1
            [10, 20]  # group 2
        ])
        hashes = hashing.apply(x)
        self.assertEqual(len(set(hashes)), 3)
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[2], hashes[3])
        self.assertNotEqual(hashes[0], hashes[2])
        self.assertNotEqual(hashes[0], hashes[4])
        self.assertNotEqual(hashes[2], hashes[4])

    def test_dimension_reduction(self):
        n_samples = 10000
        n_dim = 10
        n_hashes = 6
        hashing = RandomProjectionHashing(n_hashes, n_dim, scale_factors=5, device="cpu")
        x = torch.randn(n_samples, 10)
        hashes = hashing.apply(x)
        n_uniques_hashes = len(set(hashes))
        print(n_uniques_hashes)

        self.assertLess(n_uniques_hashes, 100)

    def test_grouping_visual_test(self):
        n_samples = 100
        n_dim = 2
        n_hashes = 15
        hashing = RandomProjectionHashing(n_hashes, n_dim, scale_factors=10, device="cpu")
        x = torch.randn(n_samples, n_dim) + torch.tensor([10, 5])
        hashes = hashing.apply(x)

        for h in set(hashes):
            i = [hi == h for hi in hashes]
            plt.plot(x[i, 0], x[i, 1], ".")
        plt.show()
