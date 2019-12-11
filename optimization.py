from steepest_descent import CForwardDiff,CBackwardDiff,CCentralDiff
from PyLineSearcher import CGSSearch,CFiSearch
import numpy as np
import time

def Test2VarFun1(x):
    return (x[0]-x[1]+2*(x[0]**2)+2*x[0]*x[1]+x[1]**2)
# x* = [-1, 1.5], f(x*) = -1.25;

def Test2VarFun2(x):
    return 0.5*(100*(x[1]-x[0]**2)**2+(1-x[0])**2)
# x* = [1, 1], f(x*) = 0;

def Test2VarFun3(x):
    return -x[0]*x[1]*np.exp(-x[0]**2-x[1]**2)
# x* = [0.7071, 0.7071] or x* = [-0.7071, -0.7071], f(x*) = -0.1839;

def Test2VarFun4(x):
    return -3*x[1]/(x[0]**2+x[1]**2+1)
# x* = [0, 1], f(x*) = -1.5;

def PowellFun(x):
    f1 = x[0]+10*x[1]
    f2 = np.sqrt(5.0)*(x[2]-x[3])
    f3 = (x[1]-2*x[2])**2
    f4 = np.sqrt(10.0)*((x[0]-x[3])**2)
    return np.sqrt(f1*f1+f2*f2+f3*f3+f4*f4)
# x* = [0, 0, 0, 0], f(x*) = 0;

class CGradDecent():
    def __init__(self, costfun, x0, Gradient = 'Backward',LineSearch = 'GsS', MinNorm = 0.001, MaxIter = 1000):
        self.costfun = costfun
        self.x0 = x0
        self.Gradient = Gradient
        self.LineSearch = LineSearch
        self.MinNorm = MinNorm
        self.MaxIter = MaxIter
        self.dim = len(self.x0)

    def set_costfun(self,costfun):
        self.costfun = costfun

    def set_x0(self,x0):
        self.x0 = x0

    def set_Maxlter(self,Maxlter):
        self.MaxIter = MaxIter

    def set_MinNorm(self,MinNorm):
        self.MinNorm = MinNorm

    def RunOptimization(self):
        lr_rate = 0.1
        if self.Gradient == 'Forword':
            _Diff = CForwardDiff(self.costfun, self.x0, self.dim, eps = self.MinNorm, percent = 1e-4)
        if self.Gradient == 'Backword':
            _Diff = CBackwardDiff(self.costfun, self.x0, self.dim, eps = self.MinNorm, percent = 1e-4)
        if self.Gradient == 'Central':
            _Diff = CCentralDiff(self.costfun, self.x0, self.dim, eps = self.MinNorm, percent = 1e-4)
        if self.LineSearch == 'GsS':
            _LineSearch = CGSSearch
        if self.LineSearch == 'FiS':
            _LineSearch = CFiSearch

        for i in range(self.MaxIter):
            descent_result = list(_Diff.GetGrad(lr_rate,self.x0))
            print('descent_result',descent_result)
            d = list(map(lambda x: -x,descent_result))[:-1]
            print('iter at:', i, self.x0, self.costfun(self.x0))
            if (descent_result[-1] < self.MinNorm):
                f_value = self.costfun(self.x0)
                print('result at:',i,self.x0,f_value)
                return self.x0  
            lr_rate = _LineSearch(self.costfun, x=self.x0, d=d, eps=0.001).Runsearch(lr_rate)
            self.x0 = [self.x0[i] + lr_rate * d[i] for i in range(self.dim)]

        print('over iter:',i,self.x0,self.costfun(self.x0))


if __name__ == '__main__':
    st = time.time()
    x0 = [1,2]
    CGradDecent(Test2VarFun2, x0, Gradient = 'Central',LineSearch = 'FiS', MinNorm = 0.001, MaxIter = 150000).RunOptimization()
    print(time.time()-st)

""" test report:
11/21: central,for,back:做func3 在x0設其他點會有問題

11/28:powellfun succed :result at: 84947 [0.005683712934733274, -0.0005683687984945234, 0.0031185999161127327, 0.0031187767850291066] 5.077524971085603e-05
"""
