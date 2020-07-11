#!/bin/python3
# Author:
# Date Created: June 29, 2020
# Date Modified: July 11, 2020
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

    class Order(Enum):
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
            train_sample = sorted(rndm.sample(num_samples_loaded,
                                              int(self._train_factor * len(self._features))))
            test_sample = list(set(num_samples_loaded) ^ set(train_sample))
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
        if order is NNData.Order.RANDOM:
            self._train_pool, self._test_pool = rndm.shuffle(self._train_pool), rndm.shuffle(self._test_pool)
        self._train_pool, self._test_pool = self._train_pool, self._test_pool

    def get_one_item(self, target_set=None):
        """Return exactly one feature/label pair as a tuple."""
        if target_set is NNData.Set.TRAIN or target_set is None:
            index_val = self._train_pool.popleft()
            return self._features[index_val], self._labels[index_val]
        elif target_set is NNData.Set.TEST:
            index_val = self._test_pool.popleft()
            print(self._features[index_val], self._labels[index_val])
            return self._features[index_val], self._labels[index_val]
        return None

    def number_of_samples(self, target_set=None):
        """Returns the total number of testing examples (if target_set is NNData.Set.TEST)
        OR total number of training examples (if the target_set is NNData.Set.TRAIN)
        OR  both combined if the target_set is None"""
        if target_set is None:
            return len(self._train_pool) + len(self._test_pool)
        elif target_set is NNData.Set.TEST:
            return len(self._test_pool)
        return len(self._train_pool)

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
        return min(1, max(factor, 0))

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


class LayerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode:
    """This will be a base class that will be a starting point for
    FFBPNeurode class"""

    class Side(Enum):
        """Used to identify relationship between neurodes."""
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._reference_value = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._neighbor = {MultiLinkNode.Side.UPSTREAM: [],
                          MultiLinkNode.Side.DOWNSTREAM: []}

    def __str__(self):
        """Prints out a representation of the node in context.
        - the ID of the node and the ID's of the neighboring nodes
        upstream and downstream."""
        pass


# Temporary Test.
def check_point_one_test():
    # Mock up a network with three inputs and three outputs

    inputs = [Neurode(LayerType.INPUT) for _ in range(3)]
    outputs = [Neurode(LayerType.OUTPUT, .01) for _ in range(3)]
    if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 0:
        print("Fail - Initial reference value is not zero")
    for node in inputs:
        node.reset_neighbors(outputs, MultiLinkNode.Side.DOWNSTREAM)
    for node in outputs:
        node.reset_neighbors(inputs, MultiLinkNode.Side.UPSTREAM)
    if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 7:
        print("Fail - Final reference value is not correct")
    if not inputs[0]._reference_value[MultiLinkNode.Side.UPSTREAM] == 0:
        print("Fail - Final reference value is not correct")

    # Report data ready from each input and make sure _check_in
    # only reports True when all nodes have reported

    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 0:
        print("Fail - Initial reporting value is not zero")
    if outputs[0]._check_in(inputs[0], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 1:
        print("Fail - reporting value is not correct")
    if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
        print("Fail - reporting value is not correct")
    if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in (double fire)")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
        print("Fail - reporting value is not correct")
    if not outputs[0]._check_in(inputs[1], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned False after all nodes were"
              "checked in")

    # Report data ready from each output and make sure _check_in
    # only reports True when all nodes have reported

    if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if inputs[1]._check_in(outputs[2], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in (double fire)")
    if not inputs[1]._check_in(outputs[1], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned False after all nodes were"
              "checked in")

    # Check that learning rates were set correctly

    if not inputs[0].learning_rate == .05:
        print("Fail - default learning rate was not set")
    if not outputs[0].learning_rate == .01:
        print("Fail - specified learning rate was not set")

    # Check that weights appear random

    weight_list = list()
    for node in outputs:
        for t_node in inputs:
            if node.get_weight(t_node) in weight_list:
                print("Fail - weights do not appear to be set up properly")
            weight_list.append(node.get_weight(t_node))


if __name__ == "__main__":
    load_XOR()
    check_point_one_test()
