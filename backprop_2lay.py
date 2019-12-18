import numpy as np

class Neuron():
    """
    Provide basic functions of a neuron, including feed-forward
    and feed-backward computations
    
    """
    inp_dict = {}
    y_dict = {}
    ready_weight = {}
    def __init__(self, weights_vec, bias):
        self.weights_vec = weights_vec
        self.bias = bias

    def forward(self, inputs):
        y_r = [0.01, 0.99]  # by default temporily
        temp_err = 0
        inputs = np.array(inputs)
        for h in range(len(self.weights_vec)):
            inputs = list(self.__calculate_total_net_input(inputs , h))
            
            print(self.y_dict)
        error = [0.5 * (y_r[i] - self.y_dict['h_{}y_{}'.format(len(self.weights_vec)-1, i)])**2 for i in range(len(self.weights_vec[-1]))]
        print(error)
        self.backward(error)

    def backward(self, error):
        delta = list(self.__node_delta(error))
        for h in range(len(self.weights_vec), -1, -1):
            self.update_weights()


    def update_weights(self, lr = 0.1):
        for h in range(len(self.weights_vec), -1, -1):

    def __calculate_total_net_input(self,inp , h):
        for i in range(len(self.weights_vec[h])):
            temp_nur = 0
            for j in range(len(inp)):
                print(inp[j],self.weights_vec[h])
                temp_nur += inp[j] * self.weights_vec[h][i][j]
            temp_nur += self.bias[h][i]
            self.y_dict['h_{}y_{}'.format(h,i)] = self.__sigmoid(temp_nur)
            yield self.y_dict['h_{}y_{}'.format(h,i)]

    def __sigmoid(self, total_net_input):
        # Apply the sigmoid activation function
        return 1/(1 + np.exp(-total_net_input))

    def __node_delta(self, error):
        # Apply the node delta function
        for i in range(len(self.weights_vec[-1])):
            y_o = self.y_dict['h_{}y_{}'.format(len(self.weights_vec)-1, i)
            yield -error[i] * y_o * (1 - y_o)


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