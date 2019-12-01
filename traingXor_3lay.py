from optimization import CGradDecent
import numpy as np
import time

def _activation(x):
    return 1/(1 + np.exp(-x))

def neural_learning(x):
    training =np.array([[0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1],                       
                        [0,1,1,0,1,0,0,1]])
    w11, w12, w13, w21, w22, w23, w31, w32, w33, w41, w42, w43 =  x[0], x[1], x[2], x[4], x[5], x[6], x[8], x[9], x[10], x[12], x[13], x[14]
    b1,b2,b3,b4 = x[3], x[7], x[11],x[15]
    x_1, x_2, x_3, y_d = training[0], training[1], training[2], training[3]
    z_1, z_2, z_3 = _activation(w11*x_1 + w12*x_2 + w13*x_3 + b1),\
                    _activation(w21*x_1 + w22*x_2 + w23*x_3 + b2),\
                    _activation(w31*x_1 + w32*x_2 + w33*x_3 + b3)
    err = y_d - _activation(w41*z_1 + w42*z_2 + w43*z_3 + b4)
    err = np.linalg.norm(err)
    return err

def neural_predict(x,in1,in2,in3):
    w11, w12, w13, w21, w22, w23, w31, w32, w33, w41, w42, w43 =  x[0], x[1], x[2], x[4], x[5], x[6], x[8], x[9], x[10], x[12], x[13], x[14]
    b1,b2,b3,b4 = x[3], x[7], x[11],x[15]
    x_1 = np.array(in1)
    x_2 = np.array(in2)
    x_3 = np.array(in3)
    z_1, z_2, z_3 = _activation(w11*x_1 + w12*x_2 + w13*x_3 + b1),\
                    _activation(w21*x_1 + w22*x_2 + w23*x_3 + b2),\
                    _activation(w31*x_1 + w32*x_2 + w33*x_3 + b3)
    f = _activation(w41*z_1 + w42*z_2 + w43*z_3 + b4)
    return f
    
if __name__ == '__main__':
    st = time.time()
    x0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77]
    xpot = CGradDecent(neural_learning, x0, 16, Gradient='Backword', LineSearch = 'GsS', MinNorm = 0.0001, MaxIter = 600000).RunOptimization()
#     xpot = [5.955828125246641, 5.955825329770617, 5.955828544139004, -15.104984818286981, 8.431519448991988, 8.43147317369757,
#  8.431524425573942, -3.9828506469904776, 8.84586591573428, 8.845850750220881, 8.845868052806239, -13.109316450012306,
#   21.605390645382293, 19.98229254453061, -21.072313302683746, -9.712646459642116]
    print(xpot)
    pre_err = neural_learning(x0)
    print('pre_err: ',pre_err)
    opt_err = neural_learning(xpot)
    print('opt_err: ',opt_err)
    pre_f = neural_predict(x0,[0,0,1,1],[0,1,0,1],[1,0,1,0])
    print('pre_f:',pre_f)
    opt_f = neural_predict(xpot,[0,0,1,1],[0,1,0,1],[1,0,1,0])
    print('opt_f:',opt_f)
    print('time:',time.time()-st)

"""
1(central,MinNorm = 0.0001,FiS, opt_err: 0.0001834699948031162, opt_f: [9.99941622e-01 9.99941622e-01 5.90128375e-05 5.90128979e-05]):
xpot = [5.955828125246641, 5.955825329770617, 5.955828544139004, -15.104984818286981, 8.431519448991988, 8.43147317369757,
 8.431524425573942, -3.9828506469904776, 8.84586591573428, 8.845850750220881, 8.845868052806239, -13.109316450012306,
  21.605390645382293, 19.98229254453061, -21.072313302683746, -9.712646459642116]

"""