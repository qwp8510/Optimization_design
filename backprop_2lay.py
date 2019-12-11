import numpy as np

class Neuron():
    """
    Provide basic functions of a neuron, including feed-forward
    and feed-backward computations
    
    """
    inp_dict = {}
    z_dict = {}
    def __init__(self, weights_vec, bias):
        self.weights_vec = weights_vec
        self.bias = bias

    def forward(self, inputs):
        temp_err = 0
        inputs = np.array(inputs)
        x_1, x_2, y_d = inputs[0], inputs[1], inputs[2]
        for i in range(0, len(inputs)-1, 1):
            temp_nur = 0
            for j in range(0, len(inputs)-1, 1):
                temp_nur += inputs[j] * self.weights_vec[j]
                del self.weights_vec[0]
            temp_nur += self.bias[j]
            z_dict['z_{}'.format(i)] = self.__sigmoid(temp_nur)
            temp_err += z_dict['z_{}'.format(i)] * self.weights_vec[-(len(inputs)-1-i)]
        err = inputs[-1] - self.__sigmoid(temp_err + self.bias[-1])
        err = np.linalg.norm(err)
        return err

    def backward(self, error):
    

    def update_weights(self, lr = 0.1):
    

    def __calculate_total_net_input(self):
    

    def __sigmoid(self, total_net_input):
        # Apply the sigmoid activation function
        return 1/(1 + np.exp(-x))

    def __node_delta(self, error):
        # Apply the node delta function
