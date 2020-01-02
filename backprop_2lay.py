import numpy as np
from collections import defaultdict
from functools import reduce


class Neuron():
    """
    Provide basic functions of a neuron, including feed-forward
    and feed-backward computations
    
    """

    inp_dict = {}
    y_dict = {}
    node_list = []
    
    def __init__(self, weights_vec, bias):
        self.weights_vec = weights_vec
        self.bias = bias

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.for_inp = np.copy(self.inputs)
        node_value = self.__calculate_total_net_input()
        self.node_list.append(node_value)
        return node_value

    def backward(self, error):
        node_delta = self.__node_delta(error)
        #print('delta:',node_delta)
        return np.average(node_delta)
        
    def update_weights(self, delta, neuron_list, lr = 0.1):
        weight_list = []
        for i in range(len(self.weights_vec)-1, -1, -1):
            lr_delta = lr * delta 
            weight_list.append(np.average(self.weights_vec[i] + lr_delta * neuron_list[i]))
        update_bias = np.average(self.bias + lr_delta)
        yield weight_list
        yield update_bias
                
    def __calculate_total_net_input(self):
        node_value = 0
        for i in range(len(self.weights_vec)):
            node_value += self.for_inp * self.weights_vec[i]
        #print(node_value)
        node_value += self.bias
        return self.__sigmoid(node_value)

    def __sigmoid(self, total_net_input):
        # Apply the sigmoid activation function
        return 1/(1 + np.exp(-total_net_input))

    def __node_delta(self, error):
        # Apply the node delta function
        y_o = self.node_list[-1]
        del self.node_list[-1]
        return -error * y_o * (1 - y_o)

class NeuronLayer():
    neuron_list = []
    def __init__(self, weights_arr, bias_vec):
        self.weights_arr = weights_arr
        self.bias_vec = bias_vec
        self.update_weight_list = []
        self.update_bias_list = []

    def inspect(self):
        # print the structure of the current neuron layer
        pass

    def feed_forward(self, inputs): 
        for i in range(len(self.weights_arr)):
            self.Neuron = Neuron(self.weights_arr[i], self.bias_vec[i])
            neuron = self.Neuron.forward(inputs)
            self.neuron_list.append(neuron)
            yield neuron

    def feed_backward(self, errors):
        # feed the node deltas of the current layer to the previous one
        node_delta = 0
        for i in range(len(self.weights_arr)-1, -1, -1):
            self.Neuron = Neuron(self.weights_arr[i], self.bias_vec[i])
            node_delta += self.Neuron.backward(errors[i])
            self.update_weights(node_delta)
        del self.neuron_list[-1]
        return self.update_weight_list, self.update_bias_list
        # errors = layer_deltas for hidden layers

    def update_weights(self, node_delta, lr = 0.1):
        updated = self.Neuron.update_weights(node_delta, self.neuron_list[-1])
        self.update_weight_list.append(next(updated))
        self.update_bias_list.append(next(updated))

class NeuronNetwork():
    def __init__(self, weights_arrs, bias_arr):
        # weights_arrs is a 3D array, and bias_arr is a 2D array
        self.weights_arrs = weights_arrs
        self.bias_arr = bias_arr

    def inspect(self):
        inputs = [[0,0,0,0,1,1,1,1],
                  [0,0,1,1,0,0,1,1],
                  [0,1,0,1,0,1,0,1]]
        outputs = [0,1,1,0,1,0,0,1]

        for i in range(20000):
            self.weights_arrs, self.bias_arr = self.train(inputs, outputs)
        print(i, 'new weights_arrs:\n',self.weights_arrs,'\nbias_arr:\n',self.bias_arr)

        return self.weights_arrs, self.bias_arr

    def feed_forward(self, inputs):
        return list(self.NeuronLayer.feed_forward(inputs))

    def compute_loss(self, training_inputs, training_outputs):
        for inp in training_inputs:
            yield (training_outputs - inp)**2  * 0.5 

    def train(self, training_inputs, training_outputs, lr = 0.1):
        # Uses online learning, ie updating the weights after each training epoch
        new_weight_arrs = []
        new_bias_arrs = []
        training_arrs = []
        # print('weights: \n',self.weights_arrs)
        # print('bias: \n',self.bias_arr)
        for i in range(len(self.weights_arrs)):
            self.NeuronLayer = NeuronLayer(self.weights_arrs[i], self.bias_arr[i])
            training_inputs =  self.feed_forward(training_inputs)
            training_arrs.append(training_inputs)

        # total_error = np.linalg.norm(np.array(training_outputs) - np.average(training_inputs))
        # print(total_error)
        # if total_error < 0.0001:
        #     return new_weight_arrs, new_bias_arrs

        for j in range(len(self.weights_arrs) - 1, -1, -1):
            errors = list(self.compute_loss(training_arrs[j], training_outputs))
            self.NeuronLayer = NeuronLayer(self.weights_arrs[j], self.bias_arr[j])
            weights, biases = self.NeuronLayer.feed_backward(errors)
            new_weight_arrs.append(weights)
            new_bias_arrs.append(biases)
        new_weight_arrs.reverse()
        new_bias_arrs.reverse()

        return new_weight_arrs, new_bias_arrs

    def __update_weights(self, lr = 0.1):
        # private functions
        pass
        #self.NeuronLayer.update_weights()

    
class neural_predict():
    y_dict = {}
    def __init__(self, weights_arrs, bias_arr, inputs):
        self.weights_arrs = weights_arrs
        self.bias = bias_arr
        self.inputs = inputs
        self.output = defaultdict(list)

    def forward(self):
        inputs = np.array(self.inputs)
        for_inp = np.copy(inputs)
        for h in range(len(self.weights_arrs)-1):
            for_inp = list(self.__calculate_total_net_input(for_inp, h))
            # self.y_dict['h_-1y_{}'.format(h)] = inputs[h]
        print('forword y_dict: ', self.y_dict)

        self.__calculate_output()

    def __calculate_total_net_input(self, inp, h):
        for i in range(len(self.weights_arrs[h])):
            temp_nur = 0
            for j in range(len(inp)):
                #print(inp)
                temp_nur += inp[j] * self.weights_arrs[h][i][j]
            temp_nur += self.bias[h][i]
            self.y_dict['h_{}y_{}'.format(h,i)] = self.__sigmoid(temp_nur)
            yield self.y_dict['h_{}y_{}'.format(h,i)]

    def __calculate_output(self):
        tmp_list = []
        for i in range(len(self.weights_arrs[-1])):
            tmp_nur = 0
            for j in range(len(self.inputs)):
                tmp_nur +=  self.y_dict['h_{}y_{}'.format(len(self.weights_arrs)-2, j)] * self.weights_arrs[-1][i][j]
            tmp_nur += self.bias[-1][i]
            print(tmp_nur)
            tmp_nur = self.__sigmoid(tmp_nur)
            tmp_list.append(tmp_nur)
        print('output:',tmp_list)

    def __sigmoid(self, total_net_input):
        # Apply the sigmoid activation function
        return 1/(1 + np.exp(-total_net_input))

if __name__ == '__main__':
    weights_arrs = [
                    [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],
                    [[1.1,1.2,1.3],[1.4,1.5,1.6],[1.7,1.8,1.9],[1.3,1.2,1.1]],
                    [[2.1,2.2,2.3,2.4]]
                   ]
    bias_arr = [
                [0.1,0.2,0.3],
                [1.1,1.2,1.3,1.4],
                [2.1]
               ]

    NeuronNetwork(weights_arrs, bias_arr).inspect()
    
        

    inputs = [[0,0,1,1],[0,1,0,1],[1,0,1,0]]
    neural_predict(weights_arrs, bias_arr, inputs).forward()


