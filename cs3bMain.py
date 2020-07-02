#!/bin/python3
# Author:
# Date Created: June 29, 2020
# Date Modified: July 1, 2020
# Description: Neural Networks capstone project.

# Imports
from enum import Enum


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
        """A method that accepts percentage as a float and returns 0 if its than 0
        otherwise 1. """
        return 0 if percentage <= 0 else 1

    def load_data(self, features: list, labels: list):
        """Compares the length of the passed in lists, if they are ore no the same
        is raises an DataMismatchError, and if features is None, it sets both self._labels
        and self._features to None and just return."""
        try:
            if len(features) != len(labels):
                raise DataMismatchError
        except DataMismatchError:
            pass
        if features is None:
            self._features, self._labels = None, None
            return


def load_xor():
    pass
