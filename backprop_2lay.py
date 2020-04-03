import numpy as np
from collections import defaultdict
from functools import reduce
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
            update_weight_vec.append(self.weights_vec[i] + lr_delta)
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
        inputs = [[0,0,0,0,1,1,1,1],
                  [0,0,1,1,0,0,1,1],
                  [0,1,0,1,0,1,0,1]]
        outputs = [0,1,1,0,1,0,0,1]

        inputs = np.array(inputs).T    
        outputs = np.array(outputs)
        self.total_node_delta_arr = []
        self.sum_node_delta_arr = []

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
        return ((training_inputs[-1]**2 + training_outputs**2)**0.5)

    def train(self, training_inputs, training_outputs, lr = 10):
        # Uses online learning, ie updating the weights after each training epoch
        node_delta_arr = []
        neuron_arr = []

        for i in range(len(self.weights_arrs)):
            self.weights_arr, self.bias_vec = self.weights_arrs[i], self.bias_arr[i]
            training_inputs =  self.feed_forward(training_inputs)
            neuron_arr.append(training_inputs)
        node_delta_vec = self.compute_loss(training_inputs, training_outputs)
        self.error += node_delta_vec[-1]
        # print(node_delta_vec[-1])
        self.error2 += self.compute_euro_loss(training_inputs, training_outputs)
        self.node_delta = 0
        node_delta_vec = [node_delta_vec]
        for j, k in zip(range(len(self.weights_arrs) - 1, 0, -1), range(len(self.weights_arrs) - 1)):
            self.weights_arr = self.weights_arrs[j]
            self.neuron_vec = neuron_arr[j-1]
            node_delta_vec += list(self.feed_backward(node_delta_vec[k]))
        node_delta_vec.reverse()
        self.total_node_delta_arr.append(node_delta_vec)

    def __update_weights(self, lr = 0.1):
        updating_weight_arrs, updating_bias_arr = [],[]
        for nodeDeltaArr in self.total_node_delta_arr:
            if not self.sum_node_delta_arr:
                self.sum_node_delta_arr = nodeDeltaArr
            else:
                self.sum_node_delta_arr = list(self.twoDimOperation(self.sum_node_delta_arr, nodeDeltaArr, '+'))
        for i in range(len(self.weights_arrs)):
            self.weights_arr, self.bias_vec, self.node_delta_vec = self.weights_arrs[i], self.bias_arr[i], self.sum_node_delta_arr[i]
            u_weights_arr, u_bias_vec = self.update_weights(lr)
            updating_weight_arrs.append(u_weights_arr)
            updating_bias_arr.append(u_bias_vec)
        # print(updating_weight_arrs)
        self.weights_arrs = updating_weight_arrs
        self.bias_arr = updating_bias_arr
    
class NeuralPredict(NeuronLayer):
    def __init__(self, weights_arrs, bias_arr):
        self.weights_arrs = weights_arrs
        self.bias_arr = bias_arr
        self.result_vec = []

    def inspect(self,inputs):
        inputs = np.array(inputs).T
        for inp in inputs:
            self.train(inp)

        print(self.result_vec)

    def compute_loss(self, training_inputs, training_outputs):
        return [(training_outputs - training_inputs[-1])**2  * 0.5] 

    def train(self, training_inputs):
        # Uses online learning, ie updating the weights after each training epoch
        for i in range(len(self.weights_arrs)):
            self.weights_arr, self.bias_vec = self.weights_arrs[i], self.bias_arr[i]
            training_inputs =  list(self.feed_forward(training_inputs))

        self.result_vec.append(training_inputs[-1])

    # def forward(self):
    #     inputs = np.array(self.inputs)
    #     for_inp = np.copy(inputs)
    #     for h in range(len(self.weights_arrs)-1):
    #         for_inp = list(self.__calculate_total_net_input(for_inp, h))
    #         # self.y_dict['h_-1y_{}'.format(h)] = inputs[h]
    #     print('forword y_dict: ', self.y_dict)

    #     self.__calculate_output()

    # def __calculate_total_net_input(self, inp, h):
    #     for i in range(len(self.weights_arrs[h])):
    #         temp_nur = 0
    #         for j in range(len(inp)):
    #             #print(inp)
    #             temp_nur += inp[j] * self.weights_arrs[h][i][j]
    #         temp_nur += self.bias[h][i]
    #         self.y_dict['h_{}y_{}'.format(h,i)] = self.__sigmoid(temp_nur)
    #         yield self.y_dict['h_{}y_{}'.format(h,i)]

    # def __calculate_output(self):
    #     tmp_list = []
    #     for i in range(len(self.weights_arrs[-1])):
    #         tmp_nur = 0
    #         for j in range(len(self.inputs)):
    #             tmp_nur +=  self.y_dict['h_{}y_{}'.format(len(self.weights_arrs)-2, j)] * self.weights_arrs[-1][i][j]
    #         tmp_nur += self.bias[-1][i]
    #         print(tmp_nur)
    #         tmp_nur = self.__sigmoid(tmp_nur)
    #         tmp_list.append(tmp_nur)
    #     print('output:',tmp_list)

    # def __sigmoid(self, total_net_input):
    #     # Apply the sigmoid activation function
    #     return 1/(1 + np.exp(-total_net_input))

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
    # weights_arrs = [[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],
    #                 [[0.23,0.31,0.51]]]
    # bias_arr = [[0.1,0.2,0.3],[0.5]]

    for i in range(1000):
        weights_arrs, bias_arr, error = NeuronNetwork(weights_arrs, bias_arr).inspect()
        print('error: ', i, error)
        
        if (error < 0.0001):
            print(i, 'new weights_arrs:\n', weights_arrs, '\nbias_arr:\n', bias_arr)
            break

    inputs = [[0,0,1,1],[0,1,0,1],[1,0,1,0]]
    # NeuralPredict(weights_arrs, bias_arr).inspect(inputs)
