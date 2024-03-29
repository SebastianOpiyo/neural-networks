from assignmentOne_Two import NNData, load_XOR
from assignment_four import LayerList
from assignment_three import FFBPNeurode
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
import time


class FFBPNetwork:
    """A base class that offers training and testing,
    One that the client will directly interact with most of the time."""

    class EmptySetException(Exception):
        pass

    def __init__(self, num_inputs: int, num_outputs: int):
        self.layers = LayerList(num_inputs, num_outputs)
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._verbose = 0
        self._epochs = 0
        self._epoch_counter = 0
        self.dataset_name = ''
        self.train_scores_mean = []
        self.validation_scores_mean = []
        self.train_sizes = []

    @staticmethod
    def alter_learning_rate(new_learning_rate_value):
        FFBPNeurode.learning_rate = new_learning_rate_value

    def verbosity_print(self, value=None):
        self._verbose = value
        if self._verbose == 0:
            # Silent
            print()
        elif self._verbose == 1:
            value = ''
            for _ in range(self._epochs):
                value += '='
            print(f'[{value}]')
        elif self._verbose == 2:
            print(f'{self._epoch_counter}/{self._epochs}')

    def remove_hidden_layer(self, position=0):
        if position < 1:
            self.layers.remove_layer()
        else:
            for _ in range(position):
                self.layers.move_forward()
            self.layers.remove_layer()

    def browse_network(self):
        """Enables one to move across the network layers,
        or return the data value of the current neural layer.
        """

        print("Please enter the following options to navigate the network and print data:\n"
              "1. back/b OR 2. forward/f OR 3. current data/CD")
        browse_option = input("Enter your navigation Option:__")
        if browse_option == "back" or browse_option == "b":
            self.layers.move_back()
            print("Success! Moved a step back in the Network")
        elif browse_option == "forwards" or browse_option == "f":
            self.layers.move_forward()
            print("Success! Moved a step forward in the Network")
        elif browse_option == "current data" or browse_option == "CD":
            current_data = self.layers.get_current_data()
            print(f'Current Data is: {current_data}')
        else:
            print("Wrong Entry: Kindly follow instructions!")

    def add_hidden_layer(self, num_nodes=5, position=0):
        """Add layer immediately after the input layer if the position is zero, else
        after the position specified, input layer inclusive."""
        self.layers.reset_to_head()
        for _ in range(position):
            self.layers.move_forward()
        self.layers.add_layer(num_nodes)

    @staticmethod
    def rmse_calculator(actual, predicted):
        """The Formulae:
                       RMSE = sqrt( sum( (predicted_i - actual_i)^2 ) / total predictions)
        """
        sum_error = 0.0
        for i in range(len(actual)):
            prediction_error = predicted[i] - actual[i]
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
        return sqrt(mean_error)

    def train(self, data_set: NNData, epochs=100, verbosity=2, order=NNData.Order.RANDOM):
        """Trains the model."""
        self._verbose = 0
        self._epochs = epochs
        self._epoch_counter = 0
        rmse_list = []
        epoch_list = []
        rmse_avg = []
        if not data_set.number_of_samples(data_set.Set.TRAIN):
            raise FFBPNetwork.EmptySetException
        for epoch in range(epochs):
            data_set.prime_data(data_set.Set.TRAIN, order)
            while not data_set.pool_is_empty(data_set.Set.TRAIN):
                feature_labels_pair = data_set.get_one_item(data_set.Set.TRAIN)
                label_list = feature_labels_pair[1]
                feature_list = feature_labels_pair[0]
                expected_list = label_list
                actual_list = []
                for feature, input_node in zip(feature_list, self.layers.input_nodes):
                    input_node.set_input(feature)
                for node in self.layers.output_nodes:
                    actual_list.append(node.node_value)
                for label, actual in zip(label_list, self.layers.output_nodes):
                    actual.set_expected(label)
                try:
                    ret_rmse = FFBPNetwork.rmse_calculator(actual_list, expected_list)
                    self._epoch_counter += 1
                    rmse_list.append(ret_rmse)
                    print(f'SAMPLE: {feature_list}, EXPECTED: {label_list}, RMSE VALUE: {ret_rmse}')
                except IndexError:
                    print("Zero Division Error Due to Empty Set.")
            # report the final RMSE
            print(f'**************** FINAL REPORT *****************')
            for rmse in range(0, len(rmse_list), 100):
                epoch_list.append(rmse)
                print(f'EPOCH {rmse} = {rmse_list[rmse]}')
            print(f'**************** AVG. RMSE *****************')
            avg_mean = np.mean(rmse_list)
            rmse_avg.append(avg_mean)
            print(f'RMSE: {avg_mean}')
            self.verbosity_print(verbosity)
        self.train_scores_mean.clear()
        self.train_sizes.clear()
        self.train_sizes = list(set(epoch_list))[:20]
        self.train_scores_mean = rmse_avg[:len(self.train_sizes)]

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        """Does the testing of the dataset."""
        self._verbose = 0
        self._epoch_counter = 0
        if not data_set.number_of_samples(data_set.Set.TEST):
            raise FFBPNetwork.EmptySetException
        data_set.prime_data(data_set.Set.TEST, order)
        rmse_counter = 0
        total_rmse = 0
        rmse_collection = []
        while not data_set.pool_is_empty(data_set.Set.TEST):
            feature_labels_pair = data_set.get_one_item(data_set.Set.TEST)
            label_list = feature_labels_pair[1]
            feature_list = feature_labels_pair[0]
            expected_list = label_list
            actual_list = []
            for feature, input_node in zip(feature_list, self.layers.input_nodes):
                input_node.set_input(feature)
            for node in self.layers.output_nodes:
                actual_list.append(node.node_value)
            for label, actual in zip(label_list, self.layers.output_nodes):
                actual.set_expected(label)
            try:
                ret_rmse = FFBPNetwork.rmse_calculator(expected_list, actual_list)
                rmse_collection.append(ret_rmse)
                self._verbose += 1
                print(f'INPUT: {feature_list}, EXPECTED: {label_list}, '
                      f'OUTPUT: {actual_list}, RMSE VALUE: {ret_rmse}')
                total_rmse += ret_rmse
                rmse_counter += 1
            except IndexError:
                print("Zero Division Error Due to Empty Set.")
        print(f'**************** FINAL RMSE REPORT *****************')
        RMSE = total_rmse/float(rmse_counter)
        print(f'RMSE: {RMSE}')
        self.verbosity_print()
        self.validation_scores_mean.clear()
        self.validation_scores_mean = rmse_collection[:len(self.train_sizes)]

    def plot_learning_curve(self):
        print(len(self.train_sizes))
        print(len(self.train_scores_mean))
        plt.plot(np.array(self.train_sizes), np.array(self.train_scores_mean), label='Training error')
        plt.plot(np.array(self.train_sizes), np.array(self.validation_scores_mean), label='Validation error')
        plt.scatter(np.array(self.train_sizes), np.array(self.validation_scores_mean))
        plt.scatter(np.array(self.train_sizes), np.array(self.train_scores_mean))
        plt.ylabel('RMSE', fontsize=14)
        plt.xlabel('Training EPOCH size', fontsize=14)
        title = 'Learning curves for a ' + self.dataset_name + ' model'
        plt.title(title, fontsize=18, y=1.03)
        plt.legend()
        plt.ylim(0, 1.5)
        plt.show()

    def plot_rmse_sin_graph(self):
        plt.style.use('seaborn-whitegrid')
        plt.title(f'An RMSE Sine Curve')
        x = np.linspace(0, np.pi/2, len(self.train_scores_mean))
        y = np.sin(x)
        plt.plot(x, y, color="green", marker='o')
        # plt.scatter(x, self.train_scores_mean, color='red')
        plt.xlabel("EPOCHS")
        plt.ylabel("RMSE")
        plt.show()


def run_iris():
    print(f'Start Timer: {time.perf_counter()}')
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(4)
    network.dataset_name = "IRIS DATASET"
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
              [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2],
              [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
              [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4],
              [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
              [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1, 0.2], [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4], [5.2, 3.5, 1.5, 0.2],
              [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4],
              [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2],
              [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5, 3.5, 1.6, 0.6],
              [5.1, 3.8, 1.9, 0.4], [4.8, 3, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
              [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5, 2, 3.5, 1],
              [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
              [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
              [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3, 5, 1.7], [6, 2.9, 4.5, 1.5],
              [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1], [5.8, 2.7, 3.9, 1.2],
              [6, 2.7, 5.1, 1.6], [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3], [5.5, 2.6, 4.4, 1.2],
              [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2], [5, 2.3, 3.3, 1], [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3, 1.1], [5.7, 2.8, 4.1, 1.3],
              [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8],
              [6.5, 3, 5.8, 2.2], [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2], [6.4, 2.7, 5.3, 1.9],
              [6.8, 3, 5.5, 2.1], [5.7, 2.5, 5, 2], [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5], [6.9, 3.2, 5.7, 2.3],
              [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2], [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1],
              [7.2, 3.2, 6, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1],
              [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2], [6.4, 2.8, 5.6, 2.2],
              [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4],
              [6.4, 3.1, 5.5, 1.8], [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5],
              [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9], [6.5, 3, 5.2, 2], [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]
    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]
    data = NNData(Iris_X, Iris_Y, .7)
    print("----------------------------------------")
    print("Train")
    # network.alter_learning_rate(.8)
    network.train(data, 100, order=NNData.Order.RANDOM)
    print("----------------------------------------")
    print("Test")
    network.test(data, order=NNData.Order.SEQUENTIAL)
    network.plot_learning_curve()
    print(f'End Timer: {time.perf_counter()}')
    return data


def run_sin():
    print(f'Start Timer: {time.perf_counter()}')
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    # network.alter_learning_rate(.8)
    network.dataset_name = "SIN DATASET"
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72], [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98], [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11], [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328], [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175], [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599], [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501], [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135], [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114], [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868], [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962], [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957], [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068], [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737], [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883], [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035], [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392], [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968], [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145], [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041], [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963], [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505], [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995], [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998], [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015], [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017], [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487], [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826], [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136], [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068], [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516], [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193], [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319], [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236], [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178], [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764], [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476], [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, .1)
    print("----------------------------------------")
    print("Train")
    network.train(data, 1000, order=NNData.Order.RANDOM)
    print("----------------------------------------")
    print("Test")
    network.test(data)
    network.plot_learning_curve()
    print(f'End Timer: {time.perf_counter()}')
    return data


def run_XOR():
    print(f'Start Timer: {time.perf_counter()}')
    data = load_XOR()
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    # network.alter_learning_rate(.8)
    network.dataset_name = "XOR DATASET"
    print("----------------------------------------")
    print("Train")
    network.train(data, 1000, order=NNData.Order.RANDOM)
    print("----------------------------------------")
    print("Test")
    network.test(data, order=NNData.Order.SEQUENTIAL)
    # network.plot_learning_curve()
    network.plot_rmse_sin_graph()
    print(f'End Timer: {time.perf_counter()}')


if __name__ == '__main__':
    run_iris()
    # run_sin()
    # run_XOR()
