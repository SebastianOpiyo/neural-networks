#!/bin/python3
# Author:
# Date Created: June 29, 2020
# Date Modified: July 11, 2020
# Description: Neural Networks capstone project.

# Imports
import collections
import math
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
import random as rndm


class DataMismatchError(Exception):
    """Raise an Exception when label & example lists have different lengths."""
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

        self._train_indices = []
        self._test_indices = []
        self._train_pool = collections.deque([])
        self._test_pool = collections.deque([])
        try:
            self.load_data(features, labels)
        except (ValueError, DataMismatchError):
            self._labels = None
            self._features = None
        self.split_set()

    @staticmethod
    def percentage_limiter(factor):
        """A method that accepts percentage as a float and returns 0 if its less than 1
        otherwise 1. """
        return min(1, max(factor, 0))

    def load_data(self, features=None, labels=None):
        """Compares the length of the passed in lists, if they are not the same
        is raises a DataMismatchError, and if features is None, it sets both self._labels
        and self._features to None and return."""
        if features is None:
            features = []
            labels = []

        if len(features) != len(labels):
            raise DataMismatchError("Label and example lists have different lengths")

        try:
            self._labels = np.array(labels, dtype=float)
            self._features = np.array(features, dtype=float)
        except ValueError:
            self._features, self._labels = [], []
            raise ValueError("label and example lists must be homogeneous"
                             "and numeric list of lists. ")

    def split_set(self, new_train_factor=None):
        """Splits the dataset, so that we have one sample for testing and another for
        training based on the train_factor"""
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        total_set_size = len(self._features)
        train_set_size = math.floor(total_set_size * self._train_factor)
        self._train_indices = rndm.sample(range(total_set_size),
                                          train_set_size)
        self._test_indices = list(set(range(total_set_size)) -
                                  set(self._train_indices))
        self._train_indices.sort()
        self._test_indices.sort()

    def prime_data(self, my_set=None, order=None):
        """This method will load one or both deques to be used as indirect indices. """
        if order is None:
            order = NNData.Order.SEQUENTIAL
        if my_set is not NNData.Set.TRAIN:
            test_indices_temp = list(self._test_indices)
            if order == NNData.Order.RANDOM:
                rndm.shuffle(test_indices_temp)
            self._test_pool = collections.deque(test_indices_temp)
        if my_set is not NNData.Set.TEST:
            train_indices_temp = list(self._train_indices)
            if order == NNData.Order.RANDOM:
                rndm.shuffle(train_indices_temp)
            self._train_pool = collections.deque(train_indices_temp)

    def get_one_item(self, target_set=None):
        """Return exactly one feature/label pair as a tuple."""
        try:
            if target_set == NNData.Set.TEST:
                index = self._test_pool.popleft()
            else:
                index = self._train_pool.popleft()
            return self._features[index], self._labels[index]
        except IndexError:
            return None

    def number_of_samples(self, target_set=None):
        """Returns the total number of testing examples (if target_set is NNData.Set.TEST)
        OR total number of training examples (if the target_set is NNData.Set.TRAIN)
        OR  both combined if the target_set is None"""
        if target_set is NNData.Set.TEST:
            return len(self._test_indices)
        elif target_set is NNData.Set.TRAIN:
            return len(self._train_indices)
        else:
            return len(self._features)

    def pool_is_empty(self, target_set=None):
        """Returns true if the target set queue(self._train_pool or
        self._test_pool) is empty otherwise False"""
        if target_set is NNData.Set.TEST:
            return len(self._test_pool) == 0
        else:
            return len(self._train_pool) == 0


def load_XOR():
    XOR_X = [[0, 0], [1, 0], [0, 1], [1, 1]]
    XOR_Y = [[0], [1], [1], [0]]
    data = NNData(XOR_X, XOR_Y, 1)


class LayerType(Enum):
    INPUT = 0
    OUTPUT = 1
    HIDDEN = 2


class MultiLinkNode(ABC):
    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {side: 0 for side in self.Side}
        self._reference_value = {side: 0 for side in self.Side}
        self._neighbors = {side: [] for side in self.Side}

    def __str__(self):
        ret_str = "-->Node " + str(id(self)) + "\n"
        ret_str = ret_str + " Input Nodes:\n"
        for key in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            ret_str = ret_str + " " + str(id(key)) + "\n"
        ret_str = ret_str + " Output Nodes\n"
        for key in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            ret_str = ret_str + " " + str(id(key)) + "\n"
        return ret_str

    @abstractmethod
    def _process_new_neighbor(self, node, side: Side):
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        """Accepts nodes as a list and side as a Side enum.
        It reset/set the nodes that link into this node either upstream
        or downstream"""
        self._neighbors[side] = nodes.copy()
        for node in nodes:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = (1 << len(nodes)) - 1


class Neurode(MultiLinkNode):

    def __init__(self, node_type: LayerType, learning_rate=.05):
        super().__init__()
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}

    def _process_new_neighbor(self, node, side: MultiLinkNode.Side):
        """Called when any new neighbors are added."""
        if side is MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = rndm.random()

    def _check_in(self, node, side: MultiLinkNode.Side):
        """Called whenever a node learns that a neighboring node has information available."""
        node_index = self._neighbors[side].index(node)
        self._reporting_nodes[side] =\
            self._reporting_nodes[side] | 1 << node_index
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes = 0
            return True
        else:
            return False

    def get_weight(self, node):
        return self._weights[node]

    @property
    def node_value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate_val(self):
        return self._learning_rate

    @learning_rate_val.setter
    def learning_rate_val(self, learning_rate: float):
        self._learning_rate = learning_rate


class FFNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(LayerType.INPUT)  # =MultiLinkNode.Side.UPSTREAM
        self._my_type = my_type

    @staticmethod
    def _sigmoid(value):
        return 1.0 / (1 + np.exp(-value))

    def _calculate_values(self):
        """Calculate the weighted sum of the upstream nodes' values.
        Pass the result through self._sigmoid() and store the
        returned value into self._value"""
        self.upstream_weight = 0
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            self.upstream_weight += self.get_weight(node) * self._value(self)  # TODO: Verify the calc
        self._value = self._sigmoid(self.upstream_weight)

    def _fire_downstream(self):
        """Call data_ready_upstream on each node's downstream neighbors
        using self.
        """
        for _ in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            self.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """Upstream neurodes call this method when they have data ready."""
        check_in_value = self._check_in(node, MultiLinkNode.Side.UPSTREAM)
        if check_in_value:
            self._calculate_values()
            self._fire_downstream()


    def set_input(self, input_value):
        """Used by the client to directly set the value of an input layer neurode."""
        self._value = input_value
        for _ in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            self.data_ready_upstream(self)


class BPNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(LayerType.INPUT)
        self._my_type = my_type
        self._delta = 0

    @staticmethod
    def _sigmoid_derivative(value):
        return value * (1.0 - value)

    def _calculate_delta(self, expected_value=None):
        self._delta = (expected_value - self.node_value) *\
                      BPNeurode._sigmoid_derivative(self.node_value)

    def data_ready_downstream(self, node):
        check_in_value = self._check_in(node, MultiLinkNode.Side.DOWNSTREAM)
        if check_in_value:
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()


    def set_expected(self, expected_value):
        self._calculate_delta(expected_value)
        for _ in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            self.data_ready_downstream(self)

    def adjust_weights(self, node, adjustment):
        self._weights[node] = adjustment

    def _update_weights(self):
        for _ in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            self.adjust_weights(self)

    def _fire_upstream(self):
        for _ in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            self.data_ready_downstream(self)


class FFBPNeurode(FFNeurode, BPNeurode):

    def __init__(self, my_type):
        FFNeurode.__init__(self, my_type)
        BPNeurode.__init__(self, my_type)
        pass


def main():
    try:
        test_neurode = BPNeurode(0)
    except:
        print("Error - Cannot instaniate a BPNeurode object")
        return
    print("Testing Sigmoid Derivative")
    try:
        assert BPNeurode._sigmoid_derivative(0) == 0
        if test_neurode._sigmoid_derivative(.4) == .24:
            print("Pass")
        else:
            print("_sigmoid_derivative is not returning the correct "
                  "result")
    except:
        print("Error - Is _sigmoid_derivative named correctly, created "
              "in BPNeurode and decorated as a static method?")
    print("Testing Instance objects")
    try:
        test_neurode._learning_rate
        test_neurode._delta
        print("Pass")
    except:
        print("Error - Are all instance objects created in __init__()?")

    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFBPNeurode(LayerType.INPUT))
        hnodes.append(FFBPNeurode(LayerType.HIDDEN))
        onodes.append(FFBPNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    print("testing learning rate values")
    for node in hnodes:
        print(f"my learning rate is {node._learning_rate}")
    print("Testing check-in")
    try:
        hnodes[0]._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] = 1
        if hnodes[0]._check_in(onodes[1], MultiLinkNode.Side.DOWNSTREAM) and \
                not hnodes[1]._check_in(onodes[1],
                                        MultiLinkNode.Side.DOWNSTREAM):
            print("Pass")
        else:
            print("Error - _check_in is not responding correctly")
    except:
        print("Error - _check_in is raising an error.  Is it named correctly? "
              "Check your syntax")
    print("Testing calculate_delta on output nodes")
    try:
        onodes[0]._value = .2
        onodes[0]._calculate_delta(.5)
        if .0479 < onodes[0]._delta < .0481:
            print("Pass")
        else:
            print("Error - calculate delta is not returning the correct value."
                  "Check the math.")
            print("        Hint: do you have a separate process for hidden "
                  "nodes vs output nodes?")
    except:
        print("Error - calculate_delta is raising an error.  Is it named "
              "correctly?  Check your syntax")
    print("Testing calculate_delta on hidden nodes")
    try:
        onodes[0]._delta = .2
        onodes[1]._delta = .1
        onodes[0]._weights[hnodes[0]] = .4
        onodes[1]._weights[hnodes[0]] = .6
        hnodes[0]._value = .3
        hnodes[0]._calculate_delta()
        if .02939 < hnodes[0]._delta < .02941:
            print("Pass")
        else:
            print("Error - calculate delta is not returning the correct value.  "
                  "Check the math.")
            print("        Hint: do you have a separate process for hidden "
                  "nodes vs output nodes?")
    except:
        print("Error - calculate_delta is raising an error.  Is it named correctly?  Check your syntax")
    try:
        print("Testing update_weights")
        hnodes[0]._update_weights()
        if onodes[0]._learning_rate == .05:
            if .4 + .06 * onodes[0]._learning_rate - .001 < \
                    onodes[0]._weights[hnodes[0]] < \
                    .4 + .06 * onodes[0]._learning_rate + .001:
                print("Pass")
            else:
                print("Error - weights not updated correctly.  "
                      "If all other methods passed, check update_weights")
        else:
            print("Error - Learning rate should be .05, please verify")
    except:
        print("Error - update_weights is raising an error.  Is it named "
              "correctly?  Check your syntax")
    print("All that looks good.  Trying to train a trivial dataset "
          "on our network")
    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFBPNeurode(LayerType.INPUT))
        hnodes.append(FFBPNeurode(LayerType.HIDDEN))
        onodes.append(FFBPNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    inodes[0].set_input(1)
    inodes[1].set_input(0)
    value1 = onodes[0]._value
    value2 = onodes[1]._value
    onodes[0].set_expected(0)
    onodes[1].set_expected(1)
    inodes[0].set_input(1)
    inodes[1].set_input(0)
    value1a = onodes[0]._value
    value2a = onodes[1]._value
    if (value1 - value1a > 0) and (value2a - value2 > 0):
        print("Pass - Learning was done!")
    else:
        print("Fail - the network did not make progress.")
        print("If you hit a wall, be sure to seek help in the discussion "
              "forum, from the instructor and from the tutors")


if __name__ == "__main__":
    main()

# Temporary Test.
# def check_point_one_test():
#     # Mock up a network with three inputs and three outputs
#     inputs = [Neurode(LayerType.INPUT) for _ in range(3)]
#     outputs = [Neurode(LayerType.OUTPUT, .01) for _ in range(3)]
#     if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 0:
#         print("Fail - Initial reference value is not zero")
#     for node in inputs:
#         node.reset_neighbors(outputs, MultiLinkNode.Side.DOWNSTREAM)
#     for node in outputs:
#         node.reset_neighbors(inputs, MultiLinkNode.Side.UPSTREAM)
#     if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 7:
#         print("Fail - Final reference value is not correct")
#     if not inputs[0]._reference_value[MultiLinkNode.Side.UPSTREAM] == 0:
#         print("Fail - Final reference value is not correct")
#
#     # Report data ready from each input and make sure _check_in
#     # only reports True when all nodes have reported
#
#     if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 0:
#         print("Fail - Initial reporting value is not zero")
#     if outputs[0]._check_in(inputs[0], MultiLinkNode.Side.UPSTREAM):
#         print("Fail - _check_in returned True but not all nodes were"
#               "checked in")
#     if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 1:
#         print("Fail - reporting value is not correct")
#     if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
#         print("Fail - _check_in returned True but not all nodes were"
#               "checked in")
#     if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
#         print("Fail - reporting value is not correct")
#     if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
#         print("Fail - _check_in returned True but not all nodes were"
#               "checked in (double fire)")
#     if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
#         print("Fail - reporting value is not correct")
#     if not outputs[0]._check_in(inputs[1], MultiLinkNode.Side.UPSTREAM):
#         print("Fail - _check_in returned False after all nodes were"
#               "checked in")
#
#     # Report data ready from each output and make sure _check_in
#     # only reports True when all nodes have reported
#
#     if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
#         print("Fail - _check_in returned True but not all nodes were"
#               "checked in")
#     if inputs[1]._check_in(outputs[2], MultiLinkNode.Side.DOWNSTREAM):
#         print("Fail - _check_in returned True but not all nodes were"
#               "checked in")
#     if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
#         print("Fail - _check_in returned True but not all nodes were"
#               "checked in (double fire)")
#     if not inputs[1]._check_in(outputs[1], MultiLinkNode.Side.DOWNSTREAM):
#         print("Fail - _check_in returned False after all nodes were"
#               "checked in")
#
#     # Check that learning rates were set correctly
#
#     if not inputs[0]._learning_rate == .05:
#         print("Fail - default learning rate was not set")
#     if not outputs[0]._learning_rate == .01:
#         print("Fail - specified learning rate was not set")
#
#     # Check that weights appear random
#
#     weight_list = list()
#     for node in outputs:
#         for t_node in inputs:
#             if node.get_weight(t_node) in weight_list:
#                 print("Fail - weights do not appear to be set up properly")
#             weight_list.append(node.get_weight(t_node))


# def check_point_two_test():
#     inodes = []
#     hnodes = []
#     onodes = []
#     for k in range(2):
#         inodes.append(FFNeurode(LayerType.INPUT))
#     for k in range(2):
#         hnodes.append(FFNeurode(LayerType.HIDDEN))
#     onodes.append(FFNeurode(LayerType.OUTPUT))
#     for node in inodes:
#         node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
#     for node in hnodes:
#         node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
#         node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
#     for node in onodes:
#         node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
#     try:
#         inodes[1].set_input(1)
#         assert onodes[0]._value == 0
#     except:
#         print("Error: Neurodes may be firing before receiving all input")
#     inodes[0].set_input(0)
#
#     # Since input node 0 has value of 0 and input node 1 has value of
#     # one, the value of the hidden layers should be the sigmoid of the
#     # weight out of input node 1.
#
#     value_0 = (1 / (1 + np.exp(-hnodes[0]._weights[inodes[1]])))
#     value_1 = (1 / (1 + np.exp(-hnodes[1]._weights[inodes[1]])))
#     inter = onodes[0]._weights[hnodes[0]] * value_0 + \
#             onodes[0]._weights[hnodes[1]] * value_1
#     final = (1 / (1 + np.exp(-inter)))
#     try:
#         print(final, onodes[0]._value)
#         assert final == onodes[0]._value
#         assert 0 < final < 1
#     except:
#         print("Error: Calculation of neurode value may be incorrect")

# if __name__ == "__main__":
#     load_XOR()
#     # check_point_one_test()
#     check_point_two_test()
