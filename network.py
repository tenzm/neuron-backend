import pickle
import json
import numpy
import scipy.special
import matplotlib.pyplot

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

    def activation_function(self, x):
        return scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0-final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


def load_training_dataset():
    data_file = open("datasets/mnist_train_100.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()
    return data_list

def train_neuron():

    input_nodes = 784
    hidden_nodes = 500
    output_nodes = 10

    # learning rate
    learning_rate = 0.2

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    training_data_file = open("datasets/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    count = 0
    epochs = 7
    for e in range(epochs):
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            count += 1
            print("Trained: " + str(count) + " / " + str(epochs * 60000))

    pickle.dump(n, open('neuron_network_values.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved")

def main():
    """
    n = pickle.load(open('neuron_network_values.pkl', 'rb'))

    test_data_file = open("datasets/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass

        scorecard_array = numpy.asarray(scorecard)
        print("performance = ", scorecard_array.sum() / scorecard_array.size)"""
    train_neuron()

if __name__ == "__main__":
    main()