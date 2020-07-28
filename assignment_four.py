from assignment_three import LayerType, FFBPNeurode


class DLLNode:
    """Node class for a DoublyLinkedList - not designed for general clients, so
    no accessors or exception raising."""

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
        self._curr = self._head
        if self._curr == self._tail:
            raise IndexError
        try:
            if self._curr is None:
                return None
            else:
                self._curr = self._curr.next
            if self._curr is None:
                return None
            else:
                self._head = self._curr
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
        self._head.next = self._tail

    def add_after_cur(self, data):
        if self._curr == self._tail:
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


class LayerList(DoublyLinkedList):
    """An iterator for the DoublyLinkedList"""

    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.input_nodes_list = []
        self.output_nodes_list = []
        self.hidden_nodes_list = []

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
        for node in range(num_nodes):
            self.hidden_nodes_list.append(FFBPNeurode(LayerType.HIDDEN))
        self.add_after_cur(self.hidden_nodes_list)

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
    # print("Pass")
    # check that each has the right number of connections
    for node in inputs:
        assert len(node._neighbors[MultiLinkNode.Side.DOWNSTREAM]) == 4
    for node in outputs:
        assert len(node._neighbors[MultiLinkNode.Side.UPSTREAM]) == 2
    # print("Pass")
    # check that the connections go to the right place
    for node in inputs:
        out_set = set(node._neighbors[MultiLinkNode.Side.DOWNSTREAM])
        check_set = set(outputs)
        assert out_set == check_set
    for node in outputs:
        in_set = set(node._neighbors[MultiLinkNode.Side.UPSTREAM])
        check_set = set(inputs)
        assert in_set == check_set
    # print("Pass")
    # add a couple layers and check that they arrived in the right order, and that iterate and rev_iterate work
    my_list.reset_to_head()
    my_list.add_layer(3)
    my_list.add_layer(6)
    print(len(my_list.get_current_data()))
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