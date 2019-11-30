from optimization import CGradDecent
import numpy as np

def _activation(x):
    return 1/(1 + np.exp(-x))

def neural_learning(x):
    training =np.array([[0,0,1,1],
                        [0,1,0,1],
                        [0,1,1,0]])
    print(x)
    w11, w12, w21, w22, w31, w32 =  x[0], x[1], x[3], x[4], x[6], x[7]
    b1,b2,b3 = x[2], x[5], x[8]
    x_1, x_2, y_d = training[0], training[1], training[2]
    z_1, z_2 = _activation(w11*x_1 + w12*x_2 + b1), _activation(w21*x_1 + w22*x_2 + b2)
    err = y_d - _activation(w31*z_1 + w32*z_2 + b3)
    err = np.linalg.norm(err)
    return err

def neural_predict(x,in1,in2):
    w11, w12, w21, w22, w31, w32 =  x[0], x[1], x[3], x[4], x[6], x[7]
    b1,b2,b3 = x[2], x[5], x[8]
    x_1 = in1
    x_2 = in2
    z_1, z_2 = _activation(w11*x_1 + w12*x_2 + b1), _activation(w21*x_1 + w22*x_2 + b2)
    f = _activation(w31*z_1 + w32*z_2 + b3)
    
if __name__ == '__main__':
    x0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #xpot = CGradDecent(neural_learning, x0, 9, Gradient='Backword', LineSearch = 'FiS', MinNorm = 0.001, MaxIter = 300000).RunOptimization()
    xpot = [5.952103209792352, 5.9524395864558475, -9.09774210558078, 7.705625970641912, 7.706927540586605, -3.5585827338159723, 
 -14.73257749680527, 14.11424426352205, -6.746860785732408]
    neural_learning(x0)
    print(xpot)
    neural_learning(xpot)

"""
[5.952103209792352, 5.9524395864558475, -9.09774210558078, 7.705625970641912, 7.706927540586605, -3.5585827338159723, 
 -14.73257749680527, 14.11424426352205, -6.746860785732408] 0.003048874543021535

 """




