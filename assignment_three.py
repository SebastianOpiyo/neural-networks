from enum import Enum
from abc import ABC, abstractmethod
import random as rndm
import numpy as np


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
