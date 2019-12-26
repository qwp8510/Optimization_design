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
    
    def __init__(self, weights_vec, bias):
        self.weights_vec = weights_vec
        self.bias = bias
        self.delta = {}
        self.update_weight_dict = defaultdict(list)
        self.update_bias_dict = defaultdict(list)

    def forward(self, inputs):
        error = []
        y_r = [0.01, 0.99]  # by default temporily
        error_total = 0
        inputs = np.array(inputs)
        for_inp = np.copy(inputs)
        print(for_inp)
        for h in range(len(self.weights_vec)):
            for_inp = list(self.__calculate_total_net_input(for_inp , h))
            self.y_dict['h_-1y_{}'.format(h)] = inputs[h]
        # print('forword y_dict: ', self.y_dict)

        
        for i in range(len(self.weights_vec[-1])):
            tmp_error = 0.5 * (y_r[i] - self.y_dict['h_{}y_{}'.format(len(self.weights_vec)-1, i)])**2
            error.append(tmp_error)
            error_total += tmp_error

        print('error: ', error_total, error)
        self.backward(error)
        return self.update_weight_dict, self.update_bias_dict

    def backward(self, error):
        for h in range(len(self.weights_vec), 0, -1):
            self.__node_delta(h, error)
            self.update_weights(h)
            #print(self.update_weight_dict)
        print('\ndelta:', self.delta)
        self.update_weight_dict = reduce(lambda a, b: a + b, self.update_weight_dict.values())
        self.update_bias_dict = reduce(lambda a, b: a + b, self.update_bias_dict.values())
        self.update_weight_dict.reverse()
        self.update_bias_dict.reverse()
        
        print('\nback update_weight:{}\nback update bias:{}'.format(self.update_weight_dict,self.update_bias_dict))

    def update_weights(self, h, lr = 0.1):
        tmp_w_dict = defaultdict(list)
        tmp_b_list = []
        for i in range(len(self.weights_vec[h-1])):
            for j in range(len(self.weights_vec[h-1][i])):
                #print("delta:::",self.delta)
                weight_element = self.weights_vec[h-1][i][j] + lr * self.delta['h_{}y_{}'.format(h, i)] * self.y_dict['h_{}y_{}'.format(h-2, j)]
                tmp_w_dict[i].append(weight_element)
            bias_element = self.bias[h-1][i] + lr * self.delta['h_{}y_{}'.format(h, i)]
            tmp_b_list.append(bias_element)

        self.update_weight_dict['h_{}'.format(h)].append(list(tmp_w_dict.values()))
        self.update_bias_dict['h_{}'.format(h)].append(tmp_b_list)
                

    def __calculate_total_net_input(self,inp , h):
        for i in range(len(self.weights_vec[h])):
            temp_nur = 0
            for j in range(len(inp)):
                temp_nur += inp[j] * self.weights_vec[h][i][j]
            temp_nur += self.bias[h][i]
            self.y_dict['h_{}y_{}'.format(h,i)] = self.__sigmoid(temp_nur)
            yield self.y_dict['h_{}y_{}'.format(h,i)]

    def __sigmoid(self, total_net_input):
        # Apply the sigmoid activation function
        return 1/(1 + np.exp(-total_net_input))

    def __node_delta(self, h, error):
        # Apply the node delta function
        if (not self.delta):
            for i in range(len(self.weights_vec[-1])):
                # print('first node_delta:', h, i)
                y_o = self.y_dict['h_{}y_{}'.format(len(self.weights_vec)-1, i)]
                self.delta['h_{}y_{}'.format(h, i)] =  (-error[i]) * y_o * (1 - y_o)
                print("delta:::",self.delta)
        else:
            for i in range(len(self.weights_vec[h][0])):
                tmp_delta = 0
                for j in range(len(self.weights_vec[h])):
                    # print(h,i,j)
                    # print(self.weights_vec[h],self.weights_vec[h][j][i])
                    tmp_delta += self.delta['h_{}y_{}'.format(h+1, j)] * self.weights_vec[h][j][i]
                    # print('temp_delta:::::',tmp_delta)
                y_o = self.y_dict['h_{}y_{}'.format(h-1, i)]
                print("y_o::::",y_o)
                tmp_delta = tmp_delta * y_o * (1 - y_o)
                self.delta['h_{}y_{}'.format(h,i)] = tmp_delta


class NeuronLayer():
    def __init__(self, weights_arr, bias_vec):
        self.weights_arr = weights_arr
        self.bias_vec = bias_vec

    def inspect(self):
        # print the structure of the current neuron layer
        pass

    def feed_forward(self, inputs): 
        for i in range(len(self.bias_vec))
            self.Neuron = Neuron(self.weights_vec, self.bias[i])
            self.feed_forward(inputs)

    def feed_backward(self, errors):
        # feed the node deltas of the current layer to the previous one


        # errors = layer_deltas for hidden layers

        pass
    def update_weights(self, lr = 0.1):
        pass

class NeuronNetwork():
    def __init__(self, weights_arrs, bias_arr):
        # weights_arrs is a 3D array, and bias_arr is a 2D array
        self.weights_arrs = weights_arrs
        self.bias_arr = bias_arr


    def inspect(self):
        inputs, outputs = 0, 0

        for i in range(100):
            self.train(inputs, outputs)

    def feed_forward(self, inputs):
        self.NeuronLayer.feed_forward(inputs)


    def compute_loss(self, training_inputs, training_outputs):
        
        pass

    def train(self, training_inputs, training_outputs, lr = 0.1):
        # Uses online learning, ie updating the weights after each training epoch
        for i in range(len(self.bias_arr)):
            self.NeuronLayer = NeuronLayer(weights_arrs, bias_arr)
            self.feed_forward(training_inputs)

        self.__update_weights()

    def __update_weights(self, lr = 0.1):
        # private functions
        self.NeuronLayer.update_weights()

if __name__ == '__main__':
    weights_arrs = [
                    [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],
                    [[1.1,1.2,1.3],[1.4,1.5,1.6],[1.7,1.8,1.9],[1.3,1.2,1.1]],
                    [[2.1,2.2,2.3,2.4],[2.5,2.6,2.7,2.8]]
                   ]
    bias_arr = [
                [0.1,0.2,0.3],
                [1.1,1.2,1.3,1.4],
                [2.1,2.2]
               ]

    for i in range(10):
        inp = [0.2,0.3,0.4]
        # print("go:", weights_arrs, '\n', bias_arr)
        weights_arrs, bias_arr= Neuron(weights_arrs, bias_arr).forward(inp)
        