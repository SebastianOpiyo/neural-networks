#!/bin/python3
# Author:
# Date Created: June 29, 2020
# Date Modified: July 1, 2020
# Description: Neural Networks capstone project.

# Imports
from enum import Enum
import numpy as np


class DataMismatchError(Exception):
    """Raised an Exception Error whenever the set sizes do not match"""
    pass


class NNData:
    """A class that enables us efficiently manage our Neural Network training and
    testing of data."""

    def __init__(self, train_factor=.9, features=None, labels=None, ):
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._labels = None
        self._features = None
        self._train_factor = NNData.percentage_limiter(train_factor)
        self.load_data(features, labels)

    class Oder(Enum):
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(percentage: float):
        """A method that accepts percentage as a float and returns 0 if its less than 1
        otherwise 1. """
        return 0 if percentage <= 0 else 1

    def load_data(self, features: list, labels: list):
        """Compares the length of the passed in lists, if they are no the same
        is raises an DataMismatchError, and if features is None, it sets both self._labels
        and self._features to None and just return."""
        try:
            if len(features) != len(labels):
                raise DataMismatchError
        except DataMismatchError:
            print("The length of the lists is a mismatch")
        if features is None:
            self._features, self._labels = None, None
            return
        try:
            if features is not None:
                self._labels = np.array(labels, dtype=float)
                self._features = np.array(features, dtype=float)
            else:
                self._labels, self._features = None, None
                raise ValueError
        except ValueError:
            print("Construction has failed, Value Error.")


def load_xor():
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    NNData(train_factor=1, features=features, labels=labels)


if __name__ == "__main__":
    load_xor()

