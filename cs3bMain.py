#!/bin/python3
# Author:
# Date Created: June 29, 2020
# Date Modified: July 5, 2020
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
        if target_set is NNData.Set.TRAIN:
            self._train_pool = self._train_indices[:]
        elif target_set is NNData.Set.TEST:
            self._test_pool = self._test_indices[:]
        else:
            self._train_pool, self._test_pool = self._train_indices[:], self._test_indices[:]
        # if order is None or NNData.Order.SEQUENTIAL, leave the pool(s) in order
        if order is NNData.Oder.RANDOM:
            rndm.shuffle(self._train_pool)
            rndm.shuffle(self._test_pool)

    def get_one_item(self, target_set=None):
        """Return exactly one feature/label pair as a tuple"""
        # If target_set is NNData.Set.TRAIN or is None, then use self._train_pool to find pair
        # If target_set is NNData.Set.TEST, use self._test_pool
        # NOTE: deques are used as indirect indices into actual example data
        # We don't return the value from deque but use value from them to return
        # the correct value from two numpy arrays
        # use "popleft()" so that the index is not reused
        # Return None if there are no indices left in the chose target_set
        pass

    def number_of_samples(self):
        """Returns the total number of testing examples (if target_set is NNData.Set.TEST)
        OR total number of training examples (if the target_set is NNData.Set.TRAIN)
        OR  both combined if the target_set is None"""
        # return total number of testing examples if target_set is NNData.Set.Test
        # OR return total number of training examples if target_set is NNData.Set.TRAIN
        # Otherwise, return both combined if the target_set is None
        pass

    def pool_is_empty(self, target_set=None):
        """Returns true if the target set queue(self._train_pool or
        self._test_pool) is empty otherwise False"""
        if target_set is None:
            target_set = self._train_pool
        return True if not target_set else False

    @staticmethod
    def percentage_limiter(factor):
        """A method that accepts percentage as a float and returns 0 if its less than 1
        otherwise 1. """
        return min(1, max(factor, 0))  # TODO:Elegant, proposed soln but how does it work?

    def load_data(self, features=None, labels=None):
        """Compares the length of the passed in lists, if they are not the same
        is raises a DataMismatchError, and if features is None, it sets both self._labels
        and self._features to None and return."""
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
def main():
    errors = False
    try:
        # Create a valid small and large dataset to be used later
        x = list(range(10))
        y = x
        our_data_0 = NNData(x, y)
        print(our_data_0._features)
        x = list(range(100))
        y = x
        our_big_data = NNData(x, y, .5)

        # Try loading lists of different sizes
        y = [1]
        try:
            our_bad_data = NNData()
            our_bad_data.load_data(x, y)
            raise Exception
        except DataMismatchError:
            pass
        except:
            raise Exception

        # Create a dataset that can be used to make sure the
        # features and labels are not confused
        x = [1, 2, 3, 4]
        y = [.1, .2, .3, .4]
        our_data_1 = NNData(x, y, .5)

    except:
        print("There are errors that likely come from __init__ or a "
              "method called by __init__")
        errors = True

    # Test split_set to make sure the correct number of examples are in
    # each set, and that the indices do not overlap.
    try:
        our_data_0.split_set(.3)
        assert len(our_data_0._train_indices) == 3
        assert len(our_data_0._test_indices) == 7
        assert (list(set(our_data_0._train_indices +
                         our_data_0._test_indices))) == list(range(10))
    except:
        print("There are errors that likely come from split_set")
        errors = True

    # Make sure prime_data sets up the deques correctly, whether
    # sequential or random.
    try:
        our_data_0.prime_data(order=NNData.Order.SEQUENTIAL)
        assert len(our_data_0._train_pool) == 3
        assert len(our_data_0._test_pool) == 7
        assert our_data_0._train_indices == list(our_data_0._train_pool)
        assert our_data_0._test_indices == list(our_data_0._test_pool)
        our_big_data.prime_data(order=NNData.Order.RANDOM)
        assert our_big_data._train_indices != list(our_big_data._train_pool)
        assert our_big_data._test_indices != list(our_big_data._test_pool)
    except:
        print("There are errors that likely come from prime_data")
        errors = True

    # Make sure get_one_item is returning the correct values, and
    # that pool_is_empty functions correctly.
    try:
        our_data_1.prime_data(order=NNData.Order.SEQUENTIAL)
        my_x_list = []
        my_y_list = []
        while not our_data_1.pool_is_empty():
            example = our_data_1.get_one_item()
            my_x_list.append(example[0])
            my_y_list.append(example[1])
        assert len(my_x_list) == 2
        assert my_x_list != my_y_list
        my_matched_x_list = [i * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        while not our_data_1.pool_is_empty(our_data_1.Set.TEST):
            example = our_data_1.get_one_item(our_data_1.Set.TEST)
            my_x_list.append(example[0])
            my_y_list.append(example[1])
        assert my_x_list != my_y_list
        my_matched_x_list = [i * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        assert set(my_x_list) == set(x)
        assert set(my_y_list) == set(y)
    except:
        print("There are errors that may come from prime_data, but could "
              "be from another method")
        errors = True

    # Summary
    if errors:
        print("You have one or more errors.  Please fix them before "
              "submitting")
    else:
        print("No errors were identified by the unit test.")
        print("You should still double check that your code meets spec.")
        print("You should also check that PyCharm does not identify any "
              "PEP-8 issues.")


if __name__ == "__main__":
    load_XOR()
    main()

