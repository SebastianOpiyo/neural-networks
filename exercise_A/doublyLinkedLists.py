#!/bin/python3
# Author:
# Date Created: July 22, 2020
# Date Modified: July 23, 2020
# Description: Neural Networks capstone project.

# Imports
from cs3bMain import FFBPNeurode, LayerType, MultiLinkNode


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

    class EmptyListError(Exception):
        """All methods should raise this error if the list is empty."""
        pass

    def move_forward(self):
        # Raise index error, if attempting to move beyond end of list.
        self._curr = self._head
        try:
            if self._curr is None:
                raise IndexError
            else:
                self._curr = self._curr.next
            if self._curr is None:
                raise IndexError
            else:
                self._head = self._curr
                # print(f'Head is at: {self._head.data}')
                return self._curr.data
        except IndexError:
            raise IndexError

    def move_back(self):
        # Raise index error, if attempting to move beyond end of list.
        self._curr = self._head
        try:
            if self._curr is None:
                raise IndexError
            else:
                self._curr = self._curr.prev
            if self._curr is None:
                raise IndexError
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
        self._head = self._curr
        # print(self._head.data)
        return self._curr.data

    # methods below take exactly one argument "data".
    def add_to_head(self, data):
        """Adds a node from the start of a list."""
        # If list is empty
        if self._head is None:
            self._head = Node(data)
            # print("New node added!")
            return
        # Add to non-empty list.
        new_node = Node(data)
        self._head.prev = new_node
        new_node.next = self._head
        new_node.prev = None
        self._head = new_node
        # print(self._head.data)
        # self.reset_to_head()

    def add_after_cur(self, data):
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


class LayerList(DoublyLinkedList, MultiLinkNode):
    """An iterator for the DoublyLinkedList"""

    def __init__(self, inputs: int, outputs: int):
        DoublyLinkedList.__init__(self)
        MultiLinkNode.__init__(self)
        self.inputs = inputs
        self.outputs = outputs
        self.input_nodes_list = []
        self.output_nodes_list = []

        for input_node in range(self.inputs):
            self.input_nodes_list.append(FFBPNeurode(LayerType.INPUT))
        MultiLinkNode.reset_neighbors(nodes=self.input_nodes_list, side=MultiLinkNode.Side.UPSTREAM)
        for output_node in range(self.outputs):
            self.output_nodes_list.append(FFBPNeurode(LayerType.OUTPUT))
        MultiLinkNode.reset_neighbors(nodes=self.output_nodes_list, side=MultiLinkNode.Side.DOWNSTREAM)

        print(self.input_nodes_list)
        print(self.output_nodes_list)

    def add_layer(self, num_nodes: int):
        """Creates a hidden layer of neurodes after the current layer
        (current linked list node.)"""
        DoublyLinkedList.add_after_cur(num_nodes)

    def remove_layer(self, data):
        """Remove a layer AFTER the current layer
        - Not allowing removal of the output layer(tail) -- raise indexError"""
        DoublyLinkedList.remove_after_cur(data)

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
    print("Pass 1")
    # check that each has the right number of connections
    for node in inputs:
        assert len(node._neighbors[MultiLinkNode.Side.DOWNSTREAM]) == 4
    for node in outputs:
        assert len(node._neighbors[MultiLinkNode.Side.UPSTREAM]) == 2
    print("Pass 2")
    # check that the connections go to the right place
    for node in inputs:
        out_set = set(node._neighbors[MultiLinkNode.Side.DOWNSTREAM])
        check_set = set(outputs)
        assert out_set == check_set
    for node in outputs:
        in_set = set(node._neighbors[MultiLinkNode.Side.UPSTREAM])
        check_set = set(inputs)
        assert in_set == check_set
    # add a couple layers and check that they arrived in the right order, and that iterate and rev_iterate work
    my_list.reset_to_head()
    my_list.add_layer(3)
    my_list.add_layer(6)
    my_list.move_forward()
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