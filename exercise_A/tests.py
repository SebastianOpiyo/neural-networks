# Temporary Test.
# def main():
#     try:
#         test_neurode = BPNeurode(0)
#     except:
#         print("Error - Cannot instaniate a BPNeurode object")
#         return
#     print("Testing Sigmoid Derivative")
#     try:
#         assert BPNeurode._sigmoid_derivative(0) == 0
#         if test_neurode._sigmoid_derivative(.4) == .24:
#             print("Pass")
#         else:
#             print("_sigmoid_derivative is not returning the correct "
#                   "result")
#     except:
#         print("Error - Is _sigmoid_derivative named correctly, created "
#               "in BPNeurode and decorated as a static method?")
#     print("Testing Instance objects")
#     try:
#         test_neurode.learning_rate
#         test_neurode.delta
#         print("Pass")
#     except:
#         print("Error - Are all instance objects created in __init__()?")
#
#     inodes = []
#     hnodes = []
#     onodes = []
#     for k in range(2):
#         inodes.append(FFBPNeurode(LayerType.INPUT))
#         hnodes.append(FFBPNeurode(LayerType.HIDDEN))
#         onodes.append(FFBPNeurode(LayerType.OUTPUT))
#     for node in inodes:
#         node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
#     for node in hnodes:
#         node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
#         node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
#     for node in onodes:
#         node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
#     print("testing learning rate values")
#     for node in hnodes:
#         print(f"my learning rate is {node.learning_rate}")
#     print("Testing check-in")
#     try:
#         hnodes[0]._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] = 1
#         if hnodes[0]._check_in(onodes[1], MultiLinkNode.Side.DOWNSTREAM) and \
#                 not hnodes[1]._check_in(onodes[1],
#                                         MultiLinkNode.Side.DOWNSTREAM):
#             print("Pass")
#         else:
#             print("Error - _check_in is not responding correctly")
#     except:
#         print("Error - _check_in is raising an error.  Is it named correctly? "
#               "Check your syntax")
#     print("Testing calculate_delta on output nodes")
#     try:
#         onodes[0]._value = .2
#         onodes[0]._calculate_delta(.5)
#         if .0479 < onodes[0]._delta < .0481:
#             print("Pass")
#         else:
#             print("Error - calculate delta is not returning the correct value."
#                   "Check the math.")
#             print("        Hint: do you have a separate process for hidden "
#                   "nodes vs output nodes?")
#     except Exception as ex:
#         print("Error - calculate_delta is raising an error.  Is it named "
#               "correctly?  Check your syntax")
#         print(repr(ex))
#     print("Testing calculate_delta on hidden nodes")
#     try:
#         onodes[0]._delta = .2
#         onodes[1]._delta = .1
#         onodes[0]._weights[hnodes[0]] = .4
#         onodes[1]._weights[hnodes[0]] = .6
#         hnodes[0]._value = .3
#         hnodes[0]._calculate_delta()
#         if .02939 < hnodes[0]._delta < .02941:
#             print("Pass")
#         else:
#             print("Error - calculate delta is not returning the correct value.  "
#                   "Check the math.")
#             print("        Hint: do you have a separate process for hidden "
#                   "nodes vs output nodes?")
#     except Exception as ex:
#         print("Error - calculate_delta is raising an error.  Is it named correctly?  Check your syntax")
#         print(repr(ex))
#     try:
#         print("Testing update_weights")
#         hnodes[0]._update_weights()
#         if onodes[0]._learning_rate == .05:
#             if .4 + .06 * onodes[0]._learning_rate - .001 < \
#                     onodes[0]._weights[hnodes[0]] < \
#                     .4 + .06 * onodes[0]._learning_rate + .001:
#                 print("Pass")
#             else:
#                 print("Error - weights not updated correctly.  "
#                       "If all other methods passed, check update_weights")
#         else:
#             print("Error - Learning rate should be .05, please verify")
#     except Exception as ex:
#         print("Error - update_weights is raising an error.  Is it named "
#               "correctly?  Check your syntax")
#         print(repr(ex))
#     print("All that looks good.  Trying to train a trivial dataset "
#           "on our network")
#     inodes = []
#     hnodes = []
#     onodes = []
#     for k in range(2):
#         inodes.append(FFBPNeurode(LayerType.INPUT))
#         hnodes.append(FFBPNeurode(LayerType.HIDDEN))
#         onodes.append(FFBPNeurode(LayerType.OUTPUT))
#     for node in inodes:
#         node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
#     for node in hnodes:
#         node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
#         node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
#     for node in onodes:
#         node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
#     inodes[0].set_input(1)
#     inodes[1].set_input(0)
#     value1 = onodes[0]._value
#     value2 = onodes[1]._value
#     onodes[0].set_expected(0)
#     onodes[1].set_expected(1)
#     inodes[0].set_input(1)
#     inodes[1].set_input(0)
#     value1a = onodes[0]._value
#     value2a = onodes[1]._value
#     if (value1 - value1a > 0) and (value2a - value2 > 0):
#         print("Pass - Learning was done!")
#     else:
#         print("Fail - the network did not make progress.")
#         print("If you hit a wall, be sure to seek help in the discussion "
#               "forum, from the instructor and from the tutors")
#
#
# if __name__ == "__main__":
#     main()

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


# if __name__ == "__main__":
#     load_XOR()
#     # check_point_one_test()
#     check_point_two_test()