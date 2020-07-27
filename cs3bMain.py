#!/bin/python3
# Author:
# Date Created: June 29, 2020
# Date Modified: July 11, 2020
# Description: Neural Networks capstone project.

# Imports
import collections
import math
import numpy as np
import random as rndm

from enum import Enum
from abc import ABC, abstractmethod


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

    def __init__(self, node_type, learning_rate=.05):
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
        self._reporting_nodes[side] = \
            self._reporting_nodes[side] | 1 << node_index
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
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
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate_val: float):
        self._learning_rate = learning_rate_val


class FFNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        return 1.0 / (1 + np.exp(-value))

    def _calculate_values(self):
        """Calculate the weighted sum of the upstream nodes' values.
        Pass the result through self._sigmoid() and store the
        returned value into self._value"""
        input_sum = 0
        for node, weight in self._weights.items():
            input_sum += node.node_value * weight
        self._value = self._sigmoid(input_sum)

    def _fire_downstream(self):
        """Call data_ready_upstream on each node's downstream neighbors
        using self.
        """
        for node in self._neighbors[Neurode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, from_node):
        """Upstream neurodes call this method when they have data ready."""
        if self._check_in(from_node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_values()
            self._fire_downstream()

    def set_input(self, input_value: float):
        """Used by the client to directly set the value of an input layer neurode."""
        self._value = input_value
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)


class BPNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @staticmethod
    def _sigmoid_derivative(value):
        return value * (1.0 - value)

    def _calculate_delta(self, expected_value=None):
        if self.node_type == LayerType.OUTPUT:
            error = expected_value - self.node_value
            self._delta = error * self._sigmoid_derivative(self.node_value)
        else:
            self._delta = 0
            for neurode in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                self._delta += neurode.get_weight(self) * neurode.delta
            self._delta *= self._sigmoid_derivative(self.node_value)

    def data_ready_downstream(self, from_node):
        if self._check_in(from_node, MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value: float):
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def adjust_weights(self, node, adjustment):
        self._weights[node] += adjustment

    def _update_weights(self):
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = self._value * node.delta * node.learning_rate
            node.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, new_value):
        self._delta = new_value


class FFBPNeurode(FFNeurode, BPNeurode):

    def __init__(self, my_type):
        FFNeurode.__init__(self, my_type)
        BPNeurode.__init__(self, my_type)
        pass


# Double Linked List Implementation.


class Node:
    """Data container and pointer initiator."""

    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    """Implements doubly linked list."""

    def __init__(self):
        self._head = None
        self._curr = None
        self._tail = None

    class EmptyListError(Exception):
        """All methods should raise this error if the list is empty."""
        pass

    def move_forward(self):
        # Raise index error, if attempting to move beyond end of list.
        self._curr = self._head
        try:
            if self._curr is None or self._curr == self._tail:
                return None
            else:
                self._curr = self._curr.next
            if self._curr is None:
                return None
            else:
                self._head = self._tail = self._curr
                return self._curr.data
        except IndexError:
            raise IndexError

    def move_back(self):
        self._curr = self._head
        try:
            if self._curr is None:
                return None
            else:
                self._curr = self._curr.prev
            if self._curr is None:
                return None
            else:
                self._head = self._curr
                # print(f'Head is at: {self._head.data}')
                return self._curr.data
        except IndexError:
            raise IndexError

    def reset_to_head(self):
        self._curr = self._head
        if self._curr is None:
            return None
        else:
            return self._curr.data

    def reset_to_tail(self):
        self._curr = self._head
        while self._curr is not None and self._curr.next is not None:
            self._curr = self._curr.next
        # print(self._head.data)
        self._tail = self._curr
        # print(self._head.data)
        return self._curr.data

    # methods below take exactly one argument "data".
    def add_to_head(self, data):
        """Adds a node from the start of a list."""
        # If list is empty
        if self._head is None and self._tail is None:
            self._head = self._tail = Node(data)
            self._head.prev = None
            self._tail.next = None
            return
        # Add to non-empty list.
        new_node = Node(data)
        self._head.prev = new_node
        new_node.next = self._head
        new_node.prev = None
        self._head = new_node
        self._tail = self._head

    def add_after_cur(self, data):
        if self._curr is self._tail:
            raise IndexError
        if self._curr is None:
            self.add_to_head(data)
            return
        new_node = Node(data)
        new_node.next = self._curr.next
        self._curr.next = new_node

    # The methods below remove a node and return its data
    # Should both raise and EmptyListError if list is empty
    def remove_from_head(self):
        try:
            if self._head is None:
                return None
            ret_val = self._head.data
            self._head = self._head.next
            self.reset_to_head()
            return ret_val
        except:
            raise DoublyLinkedList.EmptyListError

    def remove_after_cur(self):
        # Raise indexError if current node is at tail
        if self._curr == self._tail:
            raise IndexError

        if self._curr is None or self._curr.next is None:
            return None
        ret_val = self._curr.next.data
        self._curr.next = self._curr.next.next
        return ret_val

    # The method below returns the data at the current node.
    # Raise and emptyListError if list is empty.
    def get_current_data(self):
        if self._head:
            return self._head.data
        raise DoublyLinkedList.EmptyListError


# Helper code:
# doublyLinkedList1 = DoublyLinkedList()
# for item in range(3): doublyLinkedList1.add_to_head(item)
# print(doublyLinkedList1.get_current_data())
# print(doublyLinkedList1.move_forward())
# print(doublyLinkedList1.get_current_data())
# print(doublyLinkedList1.move_forward())
# print(doublyLinkedList1.get_current_data())
# print(doublyLinkedList1.move_forward())
# print(doublyLinkedList1.get_current_data())
# print(doublyLinkedList1.reset_to_tail())
# print(doublyLinkedList1.remove_from_head())
# print(doublyLinkedList1.remove_after_cur())
# print(doublyLinkedList1.add_after_cur(12))
# print(doublyLinkedList1.get_current_data())


class LayerList(DoublyLinkedList):
    """An iterator for the DoublyLinkedList"""

    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.input_nodes_list = []
        self.output_nodes_list = []

        for input_node in range(self.inputs):
            self.input_nodes_list.append(FFBPNeurode(LayerType.INPUT))

        for output_node in range(self.outputs):
            self.output_nodes_list.append(FFBPNeurode(LayerType.OUTPUT))

        for node in self.input_nodes_list:
            node.reset_neighbors(self.output_nodes_list, MultiLinkNode.Side.DOWNSTREAM)
        for node in self.output_nodes_list:
            node.reset_neighbors(self.input_nodes_list, MultiLinkNode.Side.UPSTREAM)

        self.add_to_head(self.input_nodes_list)
        self.add_after_cur(self.output_nodes_list)

    def add_layer(self, num_nodes: int):
        """Creates a hidden layer of neurodes after the current layer
        (current linked list node.)"""
        self.add_after_cur(num_nodes)

    def remove_layer(self):
        """Remove a layer AFTER the current layer
        - Not allowing removal of the output layer(tail) -- raise indexError"""
        self.remove_after_cur()

    @property
    def input_nodes(self):
        """Returns a list of input layer neurodes."""
        return self.input_nodes_list

    @property
    def output_nodes(self):
        """Returns a list of output layer neurodes."""
        return self.output_nodes_list


# Temporary test.
def layer_list_test():
    # create a LayerList with two inputs and four outputs
    my_list = LayerList(2, 4)
    # get a list of the input and output nodes, and make sure we have the right number
    inputs = my_list.input_nodes
    outputs = my_list.output_nodes
    assert len(inputs) == 2
    assert len(outputs) == 4
    print("Pass")
    # check that each has the right number of connections
    for node in inputs:
        assert len(node._neighbors[MultiLinkNode.Side.DOWNSTREAM]) == 4
    for node in outputs:
        assert len(node._neighbors[MultiLinkNode.Side.UPSTREAM]) == 2
    print("Pass")
    # check that the connections go to the right place
    for node in inputs:
        out_set = set(node._neighbors[MultiLinkNode.Side.DOWNSTREAM])
        check_set = set(outputs)
        assert out_set == check_set
    for node in outputs:
        in_set = set(node._neighbors[MultiLinkNode.Side.UPSTREAM])
        check_set = set(inputs)
        assert in_set == check_set
    print("Pass")
    # add a couple layers and check that they arrived in the right order, and that iterate and rev_iterate work
    my_list.reset_to_head()
    my_list.add_layer(3)
    my_list.add_layer(6)
    print(my_list.get_current_data()[0].node_type)
    my_list.move_forward()
    print(my_list.get_current_data()[0].node_type)
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # save this layer to make sure it gets properly removed later
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # check that information flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.node_value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.node_value
    # check that information flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    # try to remove an output layer
    try:
        my_list.remove_layer()
        assert False
    except IndexError:
        pass
    except:
        assert False
    # move and remove a hidden layer
    save_list = my_list.get_current_data()
    my_list.move_back()
    my_list.remove_layer()
    # check the order of layers again
    my_list.reset_to_head()
    assert my_list.get_current_data()[0].node_type == LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    # save a value from the removed layer to make sure it doesn't get changed
    saved_val = save_list[0].value
    # check that information still flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.node_value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.node_value
    # check that information still flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    assert saved_val == save_list[0].value


if __name__ == "__main__":
    layer_list_test()
