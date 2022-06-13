from assignment_three import LayerType, FFBPNeurode


class DLLNode:
    """Node class for a DoublyLinkedList - not designed for general clients, so
    no accessors or exception raising."""

    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    """Implements doubly linked list.
    - Behavior of Current:
    --> Make current = head when first item added
    --> Make current = next if current deleted. If current item doesn't exist,
    --> Make current = previous item.
    """

    def __init__(self):
        self._head = None
        self._current = None
        self._tail = None

    class EmptyListError(Exception):
        """Raise the exception if the list is empty."""
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self._current and self._current.next:
            ret_val = self._current.data
            self._current = self._current.next
            return ret_val
        raise StopIteration

    def move_forward(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current.next:
            self._current = self._current.next
        else:
            raise IndexError

    def move_back(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current.prev:
            self._current = self._current.prev
        else:
            raise IndexError

    def add_to_head(self, data):
        new_node = DLLNode(data)
        new_node.next = self._head
        if self._head:
            self._head.prev = new_node
        self._head = new_node
        if self._tail is None:
            self._tail = new_node
        self.reset_to_head()

    def insert_after_cur(self, data):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        new_node = DLLNode(data)
        new_node.prev = self._current
        new_node.next = self._current.next
        if self._current.next:
            self._current.next.prev = new_node
        self._current.next = new_node
        if self._tail == self._current:
            self._tail = new_node

    def remove_from_head(self):
        if not self._head:
            raise DoublyLinkedList.EmptyListError
        ret_val = self._head.data
        self._head = self._head.next
        if self._head:
            self._head.prev = None
        else:
            self._tail = None
        self.reset_to_head()
        return ret_val

    def remove_after_cur(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current == self._tail:
            raise IndexError
        ret_val = self._current.next.data
        if self._current.next == self._tail:
            self._tail = self._current
            self._current.next = None
        else:
            self._current.next = self._current.next.next
            self._current.prev = self._current
        return ret_val

    def reset_to_head(self):
        if not self._head:
            raise DoublyLinkedList.EmptyListError
        self._current = self._head

    def reset_to_tail(self):
        if not self._tail:
            raise DoublyLinkedList.EmptyListError
        self._current = self._tail

    def get_current_data(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        return self._current.data

# def dll_test():
#     my_list = DoublyLinkedList()
#     try:
#         my_list.get_current_data()
#     except DoublyLinkedList.EmptyListError:
#         print("Pass1")
#     else:
#         print("Fail")
#     for a in range(3):
#         my_list.add_to_head(a)
#     if my_list.get_current_data() != 2:
#         print("Error")
#     print("Pass2")
#     my_list.move_forward()
#     if my_list.get_current_data() != 1:
#         print("Fail!")
#     print("Pass3")
#     my_list.move_forward()
#     try:
#         my_list.move_forward()
#     except IndexError:
#         print("Pass4")
#     else:
#         print("Fail")
#     if my_list.get_current_data() != 0:
#         print("Fail")
#     my_list.move_back()
#     my_list.remove_after_cur()
#     if my_list.get_current_data() != 1:
#         print("Fail5")
#     my_list.move_back()
#     if my_list.get_current_data() != 2:
#         print("Fail")
#     try:
#         my_list.move_back()
#     except IndexError:
#         print("Pass5")
#     else:
#         print("Fail")
#     print("Pass All")
#     my_list.move_forward()
#     if my_list.get_current_data() != 1:
#         print("Fail")


class LayerList(DoublyLinkedList):
    """An iterator for the DoublyLinkedList"""

    def _link_with_next(self):
        for node in self._current.data:
            node.reset_neighbors(self._current.next.data, FFBPNeurode.Side.DOWNSTREAM)
        for node in self._current.next.data:
            node.reset_neighbors(self._current.data, FFBPNeurode.Side.UPSTREAM)

    def __init__(self, inputs, outputs):
        super().__init__()
        if inputs < 1 or outputs < 1:
            raise ValueError
        input_layer = [FFBPNeurode(LayerType.INPUT) for _ in range(inputs)]
        output_layer = [FFBPNeurode(LayerType.OUTPUT) for _ in range(outputs)]
        self.add_to_head(input_layer)
        self.insert_after_cur(output_layer)
        self._link_with_next()

    def add_layer(self, num_nodes: int):
        """Creates a hidden layer of neurodes after the current layer
        (current linked list node.)"""
        if self._current == self._tail:
            raise IndexError
        hidden_layer = [FFBPNeurode(LayerType.HIDDEN) for _ in range(num_nodes)]
        self.insert_after_cur(hidden_layer)
        self._link_with_next()
        self.move_forward()
        self._link_with_next()
        self.move_back()

    def remove_layer(self):
        """Remove a layer AFTER the current layer
        - Not allowing removal of the output layer(tail) -- raise indexError"""
        if self._current == self._tail or self._current.next == self._tail:
            raise IndexError
        self.remove_after_cur()
        self._link_with_next()

    @property
    def input_nodes(self):
        """Returns a list of input layer neurodes."""
        return self._head.data

    @property
    def output_nodes(self):
        """Returns a list of output layer neurodes."""
        return self._tail.data


# Temporary test.
def layer_list_test():
    # create a LayerList with two inputs and four outputs
    my_list = LayerList(2, 4)
    # get a list of the input and output nodes, and make sure we have the right number
    inputs = my_list.input_nodes
    outputs = my_list.output_nodes
    assert len(inputs) == 2
    assert len(outputs) == 4
    # print("Pass")
    # check that each has the right number of connections
    for node in inputs:
        assert len(node._neighbors[FFBPNeurode.Side.DOWNSTREAM]) == 4
    for node in outputs:
        assert len(node._neighbors[FFBPNeurode.Side.UPSTREAM]) == 2
    # print("Pass")
    # check that the connections go to the right place
    for node in inputs:
        out_set = set(node._neighbors[FFBPNeurode.Side.DOWNSTREAM])
        check_set = set(outputs)
        assert out_set == check_set
    for node in outputs:
        in_set = set(node._neighbors[FFBPNeurode.Side.UPSTREAM])
        check_set = set(inputs)
        assert in_set == check_set
    # print("Pass")
    # add a couple layers and check that they arrived in the right order, and that iterate and rev_iterate work
    my_list.reset_to_head()
    my_list.add_layer(3)
    my_list.add_layer(6)
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    # print("Pass")
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
    for node in inputs[1]._neighbors[FFBPNeurode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[FFBPNeurode.Side.DOWNSTREAM]):
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
    assert len(my_list.get_current_data()) == 3 # 6
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN # INPUT
    assert len(my_list.get_current_data()) == 6 # 2
    # save a value from the removed layer to make sure it doesn't get changed
    saved_val = save_list[0].node_value
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
    for node in inputs[1]._neighbors[FFBPNeurode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[FFBPNeurode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    assert saved_val == save_list[0].node_value


if __name__ == "__main__":
    layer_list_test()