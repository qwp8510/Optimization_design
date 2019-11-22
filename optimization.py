from steepest_descent import CForwardDiff,CBackwardDiff,CCentralDiff
from PyLineSearcher import CFiSearch,CGSSearch

import numpy as np

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


class CGradDecent():
    def __init__(self, costfun, x0, dim, Gradient = 'Backward',LineSearch = 'FiS', MinNorm = 0.001, MaxIter = 1000):
        self.costfun = costfun
        self.x0 = x0
        self.dim = dim
        self.Gradient = Gradient
        self.LineSearch = LineSearch
        self.MinNorm = MinNorm
        self.MaxIter = MaxIter

    def set_costfun(self,costfun):
        return costfun(self.func_x)

    def set_x0(self,x0):
        # 將x傳回function
        self.func_x = x0
        return self.set_costfun(self.costfun)

    def set_dim(self,dim):
        self.dim = dim

    def set_Maxlter(self,Maxlter):
        self.MaxIter = MaxIter

    def set_MinNorm(self,MinNorm):
        self.MinNorm = MinNorm

    def RunOptimization(self):
        lr_rate = 0.1
        if self.Gradient == 'Forword':
            Diff = CForwardDiff(self.costfun, self.x0, self.dim, eps = self.MinNorm, percent = 1e-2)
        if self.Gradient == 'Backword':
            Diff = CBackwardDiff(self.costfun, self.x0, self.dim, eps = self.MinNorm, percent = 1e-2)
        if self.Gradient == 'Central':
            Diff = CCentralDiff(self.costfun, self.x0, self.dim, eps = self.MinNorm, percent = 1e-2)

        for i in range(self.MaxIter):

            descent_result = list(Diff.GetGrad(lr_rate,self.x0))
            print('descent_result',descent_result)
            d = list(map(lambda x:-x,descent_result))[:-1]
            if (descent_result[-1] < self.MinNorm):
                f_value = self.set_x0(self.x0)
                print('result at:',i,self.x0,f_value)
                return f_value  
            lr_rate = CFiSearch(self.costfun, self.x0, d, eps=0.01).Runsearch()
            print('learing rate:',lr_rate)
            self.x0 = [self.x0[i] + lr_rate * d[i] for i in range(self.dim)]

        print('over iter:',self.x0,self.set_x0(self.x0))


if __name__ == '__main__':
    x0 = [1,2]
    dim = 2
    CGradDecent(Test2VarFun3, x0, dim, Gradient = 'Forword',LineSearch = 'FiS', MinNorm = 0.001, MaxIter = 2000).RunOptimization()

""" test report:
11/21: central,for,back:做func3 在x0設其他點會有問題

"""
