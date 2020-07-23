#!/bin/python3
# Author:
# Date Created: June 22, 2020
# Date Modified: July 23, 2020
# Description: Neural Networks capstone project.

# Imports


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
        print("Error Message: Empty list!")

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

class LayerList(DoublyLinkedList):
    """An iterator for the DoublyLinkedList"""

    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

    def add_layer(self, num_nodes: int):
        """Creates a hidden layer of neurodes after the current layer
        (current linked list node.)"""
        pass

    def remove_layer(self):
        """Remove a layer AFTER the current layer
        - Not allowing removal of the output layer(tail) -- raise indexError"""
        pass

    @property
    def input_nodes(self):
        """Returns a list of input layer neurodes."""
        pass

    @property
    def output_nodes(self):
        """Returns a list of output layer neurodes."""
        pass


# Temporary test.
def dll_test():
    my_list = DoublyLinkedList()
    try:
        my_list.get_current_data()
    except DoublyLinkedList.EmptyListError:
        print("Pass1")
    else:
        print("Fail")
    for a in range(3):
        my_list.add_to_head(a)
    if my_list.get_current_data() != 2:
        print("Error")
    print("Pass2")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail!")
    print("Pass3")
    my_list.move_forward()
    try:
        my_list.move_forward()
    except IndexError:
        print("Pass4")
    else:
        print("Fail")
    if my_list.get_current_data() != 0:
        print("Fail")
    my_list.move_back()
    my_list.remove_after_cur()
    if my_list.get_current_data() != 1:
        print("Fail5")
    my_list.move_back()
    if my_list.get_current_data() != 2:
        print("Fail")
    try:
        my_list.move_back()
    except IndexError:
        print("Pass5")
    else:
        print("Fail")
    print("Pass All")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail")


if __name__ == "__main__":
    dll_test()
    # pass
