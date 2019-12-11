class Neuron():
    def __init__(self, weights_vec, bias):
        self.weights_vec = weights_vec
        self.bias = bias

    def forward(self, inputs):
        w11, w12, w21, w22, w31, w32 =  self.weights_vec[0], self.weights_vec[1], self.weights_vec[3],\
            self.weights_vec[4], self.weights_vec[6], self.weights_vec[7]
        b1,b2,b3 = self.weights_vec[2], self.weights_vec[5], self.weights_vec[8]
        x_1, x_2, y_d = inputs[0], inputs[1], inputs[2]
        z_1, z_2 = _activation(w11*x_1 + w12*x_2 + b1), _activation(w21*x_1 + w22*x_2 + b2)
        err = y_d - _activation(w31*z_1 + w32*z_2 + b3)
        err = np.linalg.norm(err)
        print('err:',err)
        return err


    def backward(self, error):
    

    def update_weights(self, lr = 0.1):
    

    def __calculate_total_net_input(self):
    

    def __sigmoid(self, total_net_input):
        # Apply the sigmoid activation function
        return 1/(1 + np.exp(-x))

    def __node_delta(self, error):
        # Apply the node delta function
