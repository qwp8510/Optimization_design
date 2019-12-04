from optimization import CGradDecent
import numpy as np
import time

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
    print('err:',err)
    return err

def neural_predict(x,in1,in2):
    w11, w12, w21, w22, w31, w32 =  x[0], x[1], x[3], x[4], x[6], x[7]
    b1,b2,b3 = x[2], x[5], x[8]
    x_1 = np.array(in1)
    x_2 = np.array(in2)
    z_1, z_2 = _activation(w11*x_1 + w12*x_2 + b1), _activation(w21*x_1 + w22*x_2 + b2)
    f = _activation(w31*z_1 + w32*z_2 + b3)
    print('predict:',f)
    return f
    
if __name__ == '__main__':
    st = time.time()
    x0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    xpot = CGradDecent(neural_learning, x0, 9, Gradient='Forword', LineSearch = 'FiS', MinNorm = 0.001, MaxIter = 300000).RunOptimization()
#     xpot = [6.2763870796984005, 6.276624154295818, -9.586595138099005, 8.009982570820826, 8.01090375646171, 
# -3.7173554243791616, -16.022370279421505, 15.428295873785036, -7.418001687879613]
    neural_learning(x0)
    neural_learning(xpot)
    neural_predict(x0,[0,0,1,1],[0,1,0,1])
    neural_predict(xpot,[0,0,1,1],[0,1,0,1])
    print('time:',time.time()-st)

"""

1(back,MinNorm = 0.001,FiS):[5.952103209792352, 5.9524395864558475, -9.09774210558078, 7.705625970641912, 7.706927540586605, -3.5585827338159723, 
 -14.73257749680527, 14.11424426352205, -6.746860785732408] 

2(back,MinNorm = 0.001,GsS):[5.952096147464114, 5.952432525747311, -9.097731503354806, 7.70561974839969, 7.70692132488837, -3.558579461094663, 
-14.732549314051786, 14.114215477278112, -6.746846106826558] 

3(for,MinNorm = 0.001,GsS):[5.952322385702291, 5.952652828455934, -9.102801752973921, 7.705411472065045, 7.706687703780395, -3.5580603759507285, 
-14.732005396768765, 14.110007798424135, -6.747221008921933]

4(central,MinNorm = 0.001,GsS):[6.2763870796984005, 6.276624154295818, -9.586595138099005, 8.009982570820826, 8.01090375646171, -3.7173554243791616,
 -16.022370279421505, 15.428295873785036, -7.418001687879613]

5(central,MinNorm = 0.01,GsS):[0.05478019826294173, 0.15331874419042982, 0.20126234723805628, 0.3750636388489594, 0.4726939315442359,
 0.513310737336388, 0.10528897776677884, 0.09150917563799764, -0.11048945306728167]
err 0.9998576051499047

6(central,MinNorm = 0.001,GsS):[6.2763870796984005, 6.276624154295818, -9.586595138099005, 8.009982570820826, 8.01090375646171, 
-3.7173554243791616, -16.022370279421505, 15.428295873785036, -7.418001687879613]
err 0.0015182213926920065

7(central,MinNorm = 0.001,FiS):[6.276385007031306, 6.276622083283425, -9.586592033915844, 8.00997999000639, 8.010901181592404, 
-3.717354091844273, -16.022363939519728, 15.428289419644107, -7.417998377559355]
err 0.001518226924882754

8 preidct: 原:err: 1.2449576748954105,predict: [0.86040223 0.87390429 0.87033548 0.88142759]
           新:err: 0.0015182213926920065,predict: [8.63963842e-04 9.99281739e-01 9.99281696e-01 7.25742659e-04]
 """




