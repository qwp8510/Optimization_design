import numpy as np
from collections import defaultdict
from random import uniform

from utils import MatrixHandler

class Neuron(MatrixHandler):
    """
    Provide basic functions of a neuron, including feed-forward
    and feed-backward computations
    
    """
    def __init__(self, weights_vec, bias):
        self.weights_vec = weights_vec
        self.bias = bias
        self.neuron_value = 0
        self.node_delta_value = 0
        self.neuron_vec = []

    def forward(self, inputs):
        self.inputs = inputs
        node_value = self.__calculate_total_net_input()
        return node_value

    def backward(self, error):
        nodeDelta = 0
        for i, weight in enumerate(self.weights_vec):
            self.neuron_value = self.neuron_vec[i]
            node = self.__node_delta(error)
            nodeDelta = node * weight
            yield nodeDelta

    def update_weights(self, lr = 0.1):
        update_weight_vec = []
        lr_delta = lr * self.node_delta_value
        for i in range(len(self.weights_vec)):
            # print(self.weights_vec[i], self.neuron_value, self.neuron_value[i])
            print(i,lr_delta ,self.neuron_value[i])
            update_weight_vec.append(self.weights_vec[i] + lr_delta * self.neuron_value[i])
        return update_weight_vec, self.bias + lr_delta
                
    def __calculate_total_net_input(self):
        node_value = 0
        for i in range(len(self.weights_vec)):
            node_value += self.inputs[i] * self.weights_vec[i]
        node_value += self.bias
        return self.__sigmoid(node_value)

    def __sigmoid(self, total_net_input):
        # Apply the sigmoid activation function
        return 1/(1 + np.exp(-total_net_input))

    def __node_delta(self, error):
        # Apply the node delta function
        return -error * self.neuron_value * (1 - self.neuron_value)

class NeuronLayer(Neuron):
    def __init__(self, weights_arr, bias_vec):
        self.weights_arr = weights_arr
        self.bias_vec = bias_vec
        self.node_delta_vec = []

    def feed_forward(self, inputs): 
        for i in range(len(self.weights_arr)):
            self.weights_vec, self.bias = self.weights_arr[i], self.bias_vec[i]
            yield self.forward(inputs)

    def feed_backward(self, errors):
        # feed the node deltas of the current layer to the previous one
        nodeDelta = []
        for i in range(len(errors)-1, -1, -1):
            self.weights_vec = self.weights_arr[i]
            # self.neuron_value = self.neuron_vec[i]
            if not nodeDelta:
                nodeDelta = list(self.backward(errors[i]))
            else:
                nodeDelta = list(self.add_vec(nodeDelta, list(self.backward(errors[i]))))
        yield nodeDelta
       
    def update_weights(self, lr = 0.1):
        updating_weight_arr, updating_bias_vec = [] , []
        for i in range(len(self.weights_arr)):
            self.neuron_value = self.neuron_vec
            self.weights_vec, self.bias, self.node_delta_value = self.weights_arr[i], self.bias_vec[i], self.node_delta_vec[i]
            u_weights_vec, u_bias = super(NeuronLayer,self).update_weights(lr)
            updating_weight_arr.append(u_weights_vec)
            updating_bias_vec.append(u_bias)
        return updating_weight_arr, updating_bias_vec

class NeuronNetwork(NeuronLayer):
    def __init__(self, weights_arrs, bias_arr):
        # weights_arrs is a 3D array, and bias_arr is a 2D array
        self.weights_arrs = weights_arrs
        self.bias_arr = bias_arr
        self.neuronShape = weights_arrs
        self.error = 0
        self.error2 = 0

    def inspect(self):
        inputs = [[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]]
        outputs = [0,1,1,0,1,0,0,1]

        self.total_node_delta_arr = []
        self.total_neuron_arr = []

        for i in range(len(inputs)):
            self.train(inputs[i], outputs[i])
        self.__update_weights()
        # print('error_euro: ',self.error2)
        return self.weights_arrs, self.bias_arr, self.error

    def feed_forward(self, inputs):
        return list(super(NeuronNetwork, self).feed_forward(inputs))

    def compute_loss(self, training_inputs, training_outputs):
        return [((training_outputs - training_inputs[-1])**2) * 0.5] 

    def compute_euro_loss(self, training_inputs, training_outputs):
        return [((training_inputs[-1]**2 + training_outputs**2)**0.5)]

    def train(self, training_inputs, training_outputs, lr = 0.1):
        # Uses online learning, ie updating the weights after each training epoch
        inps = training_inputs
        neuron_arr = []
        for i in range(len(self.weights_arrs)):
            self.weights_arr, self.bias_vec = self.weights_arrs[i], self.bias_arr[i]
            inps =  self.feed_forward(inps)
            neuron_arr.append(inps)
        node_delta_vec = self.compute_loss(inps, training_outputs)
        # node_delta_vec = self.compute_euro_loss(training_inputs, training_outputs)
        self.error += node_delta_vec[-1]
        self.node_delta = 0
        node_delta_vec = [node_delta_vec]
        for j, k in zip(range(len(self.weights_arrs) - 1, 0, -1), range(len(self.weights_arrs) - 1)):
            self.weights_arr = self.weights_arrs[j]
            self.neuron_vec = neuron_arr[j-1]
            node_delta_vec += list(self.feed_backward(node_delta_vec[k]))
        node_delta_vec.reverse()
        neuron_arr.insert(0, list(training_inputs))
        self.total_node_delta_arr.append(node_delta_vec)
        self.total_neuron_arr.append(neuron_arr)

    def __update_weights(self, lr = 0.1):
        updating_weight_arrs, updating_bias_arr, sum_node_delta_arr, sum_neuron_arr = [], [], [], []
        for nodeDeltaArr in self.total_node_delta_arr:
            if not sum_node_delta_arr:
                sum_node_delta_arr = nodeDeltaArr
            else:
                sum_node_delta_arr = list(self.twoDimOperation(sum_node_delta_arr, nodeDeltaArr, '+'))
        for neuronArr in self.total_neuron_arr:
            if not sum_neuron_arr:
                sum_neuron_arr = neuronArr
            else:
                sum_neuron_arr = list(self.twoDimOperation(sum_neuron_arr, neuronArr, '+'))
        for i in range(len(self.weights_arrs)):
            self.neuron_vec = sum_neuron_arr[i]
            self.weights_arr, self.bias_vec, self.node_delta_vec = self.weights_arrs[i], self.bias_arr[i], sum_node_delta_arr[i]
            u_weights_arr, u_bias_vec = self.update_weights(lr)
            updating_weight_arrs.append(u_weights_arr)
            updating_bias_arr.append(u_bias_vec)
        # print(updating_weight_arrs)
        self.weights_arrs = updating_weight_arrs
        self.bias_arr = updating_bias_arr
    

def make_coefficient(inpNum, layers):
    def gen_value(num):
        for _ in range(num):
            yield uniform(-3, 3)

    def make_w(num, layer):
        for _ in range(layer):
            yield list(gen_value(num))
            
    def make_b(num):
        yield from gen_value(num)

    for layer in layers:    
        yield list(make_w(inpNum, layer)), list(make_b(layer))
        inpNum = layer

if __name__ == '__main__':
    inputNum = 3
    hiddenLayers = [3, 4, 1]
    weights_arrs, bias_arr = zip(*make_coefficient(inputNum, hiddenLayers))
    # weights_arrs = [
    #                 [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],
    #                 [[1.1,1.2,1.3],[1.4,1.5,1.6],[1.7,1.8,1.9],[1.3,1.2,1.1]],
    #                 [[2.1,2.2,2.3,2.4]]
    #                ]
    # bias_arr = [
    #             [0.1,0.2,0.3],
    #             [1.1,1.2,1.3,1.4],
    #             [2.1]
            #    ]
    # weights_arrs = [[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],
    #                 [[0.23,0.31,0.51]]]
    # bias_arr = [[0.1,0.2,0.3],[0.5]]

    for i in range(1):
        weights_arrs, bias_arr, error = NeuronNetwork(weights_arrs, bias_arr).inspect()
        print('error: ', i, error)
        
        if (error < 0.0001):
            print(i, 'new weights_arrs:\n', weights_arrs, '\nbias_arr:\n', bias_arr)
            break

    inputs = [[0,0,1,1],[0,1,0,1],[1,0,1,0]]
    # NeuralPredict(weights_arrs, bias_arr).inspect(inputs)
