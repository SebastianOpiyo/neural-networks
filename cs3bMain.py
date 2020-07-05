#!/bin/python3
# Author:
# Date Created: June 29, 2020
# Date Modified: July 1, 2020
# Description: Neural Networks capstone project.

# Imports
from enum import Enum
import numpy as np
import collections
import random as rndm


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

    def __init__(self, features=None, labels=None, train_factor=.9):
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
            self.split_set(train_factor)
        except (ValueError, DataMismatchError):
            pass

    def split_set(self, new_train_factor=None):
        """Splits the dataset, so that we have one sample for testing and another for
        training based on the train_factor"""
        if new_train_factor:
            self._train_factor = self.percentage_limiter(new_train_factor)
            num_samples_loaded = range(len(self._features))
            test_sample = sorted(rndm.sample(num_samples_loaded,
                                             int(self._train_factor * len(self._features))))
            train_sample = list(set(num_samples_loaded) ^ set(test_sample))
            self._test_indices = self._features[test_sample]
            self._train_indices = self._features[train_sample]
            return self._train_indices, self._test_indices

    def prime_data(self, target_set=None, order=None):
        """This method will load one or both deques to be used as indirect indices. """
        pass

    def get_one_item(self, target_set=None):
        """Return exactly one feature/label pair as a tuple"""
        pass

    def number_of_samples(self):
        pass

    def pool_is_empty(self, target_set=None):
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
    data = NNData(XOR_X, XOR_Y, 1)


# Temporary Test.
def unit_test():
    errors = False
    try:
        # Create a valid small and large dataset to be used later
        x = list(range(10))
        y = x
        our_data_0 = NNData(x, y)
        print("Print features for our_data_0")
        print(our_data_0._features)
        x = list(range(100))
        y = x
        our_big_data = NNData(x, y, .5)
        print()
        print("Print features for our_big_data")
        print(our_big_data._features)
        print()
        print("_train_indices and _test_indices for our_big_data with percentile=.5")
        print(our_big_data._train_indices)
        print(our_big_data._test_indices)
        our_big_data.split_set(.3)
        print()
        print("_train_indices and _test_indices for our_big_data with percentile=.3")
        print(our_big_data._train_indices)
        print(our_big_data._test_indices)
        # Try loading lists of different sizes
        y = [1]

    except(ValueError, DataMismatchError):
        print("There are errors that likely come from __init__ or a "
              "method called by __init__")
        errors = True

    # Test split_set to make sure the correct number of samples are in
    # each set, and that the indices do not overlap.
    try:
        x = list(range(10))
        y = x
        our_data_0 = NNData(x, y)
        our_data_0.split_set(.3)
        print()
        print("_train_indices and _test_indices for our_data_0 with percentile .3")
        print(our_data_0._train_indices)
        print(our_data_0._test_indices)
        assert len(our_data_0._train_indices) == 7
        assert len(our_data_0._test_indices) == 3
        # assert (list(set(our_data_0._test_indices +
        #                  our_data_0._train_indices))) == list(range(10))
    except AssertionError:
        print("There are errors that likely come from split_set")
        errors = True  # Summary
    if errors:
        print("You have one or more errors.  Please fix them before "
              "submitting")
    else:
        print("No errors were identified by the unit test.")
        print("You should still double check that your code meets spec.")
        print("You should also check that PyCharm does not identify any "
              "PEP-8 issues.")


if __name__ == "__main__":
    # load_XOR()
    unit_test()

