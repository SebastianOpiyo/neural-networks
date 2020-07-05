#!/bin/python3
# Author:
# Date Created: June 29, 2020
# Date Modified: July 1, 2020
# Description: Neural Networks capstone project.

# Imports
from enum import Enum
import numpy as np
import collections


class DataMismatchError(Exception):
    """Raise an Exception Error whenever the set sizes do not match"""
    pass


class NNData:
    """A class that enables us efficiently manage our Neural Network training and
    testing of data."""

    class Oder(Enum):
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        TRAIN = 0
        TEST = 1

    def __init__(self, train_factor=.9, features=None, labels=None):
        self._train_factor = NNData.percentage_limiter(train_factor)
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._train_indices = []
        self._test_indices = []
        self._train_pool = collections.deque([])
        self._test_pool = collections.deque([])
        self._labels = None
        self._features = None
        try:
            self.load_data(features, labels)
        except (ValueError, DataMismatchError):
            pass
        self.split_set()

    def split_set(self, new_train_factor=None):
        if new_train_factor:
            # Remember to use percentage_limiter to make sure the value stays within range
            self._train_factor = new_train_factor

    def prime_data(self, target_set=None, order=None):
        """This method will load one or both deques to be used as indirect indices. """
        pass

    def get_one_item(self, target_set=None):
        """Return exactly one feature/label pair as a tuple"""
        pass

    @staticmethod
    def percentage_limiter(factor):
        """A method that accepts percentage as a float and returns 0 if its less than 1
        otherwise 1. """
        return min(1, max(factor, 0))  # TODO:Elegant, proposed soln but how does it work?

    def load_data(self, features=None, labels=None):
        """Compares the length of the passed in lists, if they are no the same
        is raises an DataMismatchError, and if features is None, it sets both self._labels
        and self._features to None and just return."""
        if features is None or labels is None:
            self._features, self._labels = None, None
            return

        if len(features) != len(labels):
            raise DataMismatchError("Label and example lists have different lengths")

        try:
            self._labels = np.array(labels, dtype=float)
            self._features = np.array(features, dtype=float)
        except ValueError:
            self._features, self._labels = None, None
            raise ValueError("label and example lists must be homogeneous"
                             "and numeric list of lists. ")


def load_XOR():
    XOR_X = [[0, 0], [1, 0], [0, 1], [1, 1]]
    XOR_Y = [[0], [1], [1], [0]]
    data = NNData(1, XOR_X, XOR_Y)

# Temporary Test.


if __name__ == "__main__":
    load_xor()

