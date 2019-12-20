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
    update_weight_dict = defaultdict(list)
    delta = {}
    def __init__(self, weights_vec, bias):
        self.weights_vec = weights_vec
        self.bias = bias

    def forward(self, inputs):
        y_r = [0.01, 0.99]  # by default temporily
        temp_err = 0
        inputs = np.array(inputs)
        for_inp = np.copy(inputs)
        for h in range(len(self.weights_vec)):
            for_inp = list(self.__calculate_total_net_input(for_inp , h))
            
            self.y_dict['h_-1y_{}'.format(h)] = inputs[h]
            print('forword y_dict: ', self.y_dict)
        error = [0.5 * (y_r[i] - self.y_dict['h_{}y_{}'.format(len(self.weights_vec)-1, i)])**2 for i in range(len(self.weights_vec[-1]))]
        print('error: ', error)
        self.backward(error)

    def backward(self, error):
        
        for h in range(len(self.weights_vec), 0, -1):
            self.__node_delta(h, error)
            self.update_weights(h)
            #print(self.update_weight_dict)
        print('\ndelta:', self.delta)
        self.update_weight_dict = reduce(lambda a, b: a + b, self.update_weight_dict.values())
        self.update_weight_dict.reverse()
        
        print('\nback update_weight: ', self.update_weight_dict)


    def update_weights(self, h, lr = 0.1):
        tmp_dict = defaultdict(list)
        for i in range(len(self.weights_vec[h-1])):
            for j in range(len(self.weights_vec[h-1][i])):
                print("loc: ", h, i, j)
                #print("delta:::",self.delta)
                weight = self.weights_vec[h-1][i][j] + lr * self.delta['h_{}y_{}'.format(h, i)] * self.y_dict['h_{}y_{}'.format(h-2, j)]
                tmp_dict[i].append(weight)
        self.update_weight_dict['h_{}'.format(h)].append(list(tmp_dict.values()))
                

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
                y_o = self.y_dict['h_{}y_{}'.format(len(self.weights_vec)-1, i)]
                self.delta['h_{}y_{}'.format(h, i)] =  (-error[i]) * y_o * (1 - y_o)
        else:
            for i in range(len(self.weights_vec[h][0])):
                tmp_delta = 0
                for j in range(len(self.weights_vec[h])):
                    # print(h,i,j)
                    # print(self.weights_vec[h],self.weights_vec[h][j][i])
                    tmp_delta = self.delta['h_{}y_{}'.format(h+1, j)] * self.weights_vec[h][j][i]
                y_o = self.y_dict['h_{}y_{}'.format(h-1, i)]
                tmp_delta = tmp_delta * y_o * (1 - y_o)
                self.delta['h_{}y_{}'.format(h,i)] = tmp_delta



if __name__ == '__main__':
    weights_arrs = [[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],
                    [[1.1,1.2,1.3],[1.4,1.5,1.6],[1.7,1.8,1.9],[1.3,1.2,1.1]],
                    [[2.1,2.2,2.3,2.4],[2.5,2.6,2.7,2.8]]
                   ]
    bias_arr = [[0.1,0.2,0.3],
                [1.1,1.2,1.3,1.4],
                [2.1,2.2,]
               ]
    inp = [0.2,0.3,0.4]
    
    Neuron(weights_arrs, bias_arr).forward(inp)