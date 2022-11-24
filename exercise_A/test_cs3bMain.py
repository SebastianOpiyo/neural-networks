#!/bin/python3
# Author:
# Date Created: June 29, 2020
# Date Modified: July 2, 2020
# Description: Neural Networks capstone project unittest test suite.

import unittest
from cs3bMain import NNData as Nnd
from cs3bMain import DataMismatchError as DataME


class TestNNData(unittest.TestCase):
    """T.D.D for the NNData Class, testing the following:
    1. NNData.load_data() raises DataMismatchError if features and labels
       have different lengths when calling and verifying that self._features and
       self._labels are set to None.
    2. NNData.load_data() raises ValueError if features or labels contain non-float
       values, & that they are set to none.
    3. Verify that if invalid lists are passed to the constructor e.g lists of
       of different length or non-homogeneous, sets self._labels & self._features to None
    4. Verify that NNData limits the training factor to 0 if -ve number is passed.
    5. Verify that NNData limits training factor to one if num greater than 1 is passed.
    """

    def setUp(self):
        self._nndata = Nnd(train_factor=.9, labels=None, features=None)
        self.DataMismatch = DataME

    def test_lists_length_equality_and_wrongInputValue(self):
        """Test for list length equality and for non-float input values."""
        features = [2, 4, 6, 7]
        labels = [2, 3, 5]
        with self.assertRaises(self.DataMismatch):
            self._nndata.load_data(features=features, labels=labels)
        with self.assertRaises(ValueError):
            self._nndata.load_data(features=features, labels=labels)

    def test_return_None_with_invalid_data(self):
        """Testing for None return. """
        features, labels = [2, 4, 6, 7], [2, 3, 5]
        self.assertIsNone(self._nndata.load_data(features=features, labels=labels), msg="The Lists are not equal!")
        features, labels = [2, 4, 6, 7], [2, "foo", 12, "Dola", 3]
        self.assertIsNone(self._nndata.load_data(features=features, labels=labels), msg="Invalid Data in the lists!")

    def test_training_factor_negative_positive_check(self):
        """Testing for negative values and those greater than 1"""
        self.assertEqual(0, self._nndata.percentage_limiter(-.3))
        self.assertEqual(1, self._nndata.percentage_limiter(10.0))


if __name__ == "__main__":
    unittest.main()

# Temporary test for reference.
"""
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
        # Broadcast issue...probably in need of reshape TODO: Still looking at broadcast, I haven't grasped the idea.
        # assert (list(set(our_data_0._test_indices +
        #                   our_data_0._train_indices))) == list(range(10))
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
"""