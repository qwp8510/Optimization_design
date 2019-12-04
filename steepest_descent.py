import numpy as np
from PyLineSearcher import CFiSearch,CGSSearch

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

def test_fun(x):
    return x[0]**2 +x[1]
# x*=(2,1)

# Forward diff
x_value = [-2,2]

class CForwardDiff():
    descent_value_cols = {}
    for_value_cols = {}
    forword_value_cols = {}
    def __init__(self, costfun, x, dim, eps = 1e-4, percent = 1e-4):
        self.costfun = costfun
        self.x = x
        self.dim = dim
        self.eps = eps
        self.percent = percent

    def set_costfun(self, costfun):
        self.costfun = costfun

    def set_x(self, x):
        self.x = x

    def set_dim(self, dim):
        self.dim = dim

    def set_eps(self, eps):
        self.eps = eps

    def set_percent(self, percent):
        self.percent = percent

    def GetGrad(self,step_size,x0):
        self.x = x0
        return self.Forword_diff()
    
    def Forword_diff(self):
        #計算需變動後的值
        forword_result = 0 # initial
        for i in range(self.dim):
            self.descent_value_cols['descent_value_{}'.format(i)] = self.percent * self.x[i] + self.eps
        for i in range(self.dim):    
            self.for_value_cols['for_x_{}'.format(i)] = [val + self.descent_value_cols['descent_value_{}'.format(i)]\
                                                           if i==j else val for j,val in enumerate(self.x)]
            self.forword_value_cols['forword_x_{}'.format(i)] = (self.costfun(self.for_value_cols['for_x_{}'.format(i)]) - \
                                                                    self.costfun(self.x)) / self.descent_value_cols['descent_value_{}'.format(i)]
            forword_result += self.forword_value_cols['forword_x_{}'.format(i)]**2
            yield self.forword_value_cols['forword_x_{}'.format(i)]
        print('forword_result:',forword_result**0.5)
        yield forword_result**0.5
 
        
class CBackwardDiff():
    descent_value_cols = {}
    back_value_cols = {}
    backword_value_cols = {}
    def __init__(self, costfun, x, dim, eps = 1e-4, percent = 1e-4):
        self.costfun = costfun
        self.x = x
        self.dim = dim
        self.eps = eps
        self.percent = percent

    def set_costfun(self, costfun):
        self.costfun = costfun

    def set_x(self, x):
        self.x = x

    def set_dim(self, dim):
        self.dim = dim

    def set_eps(self, eps):
        self.eps = eps

    def set_percent(self, percent):
        self.percent = percent

    def GetGrad(self,step_size,x0):
        self.x = x0
        return self.Backword_diff()

    def Backword_diff(self):
        backword_result = 0 # initial
        descent_value_cols = ['descent_value_{}'.format(i) for i in range(self.dim)]
        for i in range(self.dim):
            self.descent_value_cols['descent_value_{}'.format(i)] = self.percent * self.x[i] + self.eps 
        for i in range(self.dim):    
            self.back_value_cols['back_x_{}'.format(i)] = [val - self.descent_value_cols['descent_value_{}'.format(i)]\
                                                           if i==j else val for j,val in enumerate(self.x)]
            self.backword_value_cols['backword_x_{}'.format(i)] = (self.costfun(self.x) - self.costfun(self.back_value_cols['back_x_{}'.format(i)]))\
                                                                     / self.descent_value_cols['descent_value_{}'.format(i)]     
            backword_result += self.backword_value_cols['backword_x_{}'.format(i)]**2
            yield self.backword_value_cols['backword_x_{}'.format(i)]
        print('backword_result',backword_result**0.5)
        yield backword_result**0.5


class CCentralDiff():
    descent_value_cols = {}
    for_value_cols = {}
    back_value_cols = {}
    central_value_cols = {}
    def __init__(self, costfun, x, dim, eps = 1e-4, percent = 1e-4):
        self.costfun = costfun
        self.x = x
        self.dim = dim
        self.eps = eps
        self.percent = percent

    def set_costfun(self, costfun):
        self.costfun = costfun

    def set_x(self, x):
        self.x = x

    def set_dim(self, dim):
        self.dim = dim

    def set_eps(self, eps):
        self.eps = eps

    def set_percent(self, percent):
        self.percent = percent

    def GetGrad(self,step_size,x0):
        self.x = x0
        return self.Central_diff()

    def Central_diff(self):
        central_result = 0 # initial
        for i in range(self.dim):
            self.descent_value_cols['descent_value_{}'.format(i)] = self.percent * self.x[i] + self.eps 
        for i in range(self.dim):   
            self.for_value_cols['for_x_{}'.format(i)] = [val + self.descent_value_cols['descent_value_{}'.format(i)]\
                                                           if i==j else val for j,val in enumerate(self.x)]
            self.back_value_cols['back_x_{}'.format(i)] = [val - self.descent_value_cols['descent_value_{}'.format(i)]\
                                                           if i==j else val for j,val in enumerate(self.x)]   

            self.central_value_cols['central_x_{}'.format(i)] = (self.costfun(self.for_value_cols['for_x_{}'.format(i)]) - \
                                                                    self.costfun(self.back_value_cols['back_x_{}'.format(i)])) / (self.descent_value_cols['descent_value_{}'.format(i)])
            central_result += self.central_value_cols['central_x_{}'.format(i)]**2
            yield self.central_value_cols['central_x_{}'.format(i)]
        print('central_result',central_result**0.5)

        yield central_result**0.5


# if __name__ == "__main__":
#     CForwardDiff(Test2VarFun4, x_value, dim=2, eps = 1e-3, percent = 1e-2).GetGrad(0.1)
#     CBackwardDiff(Test2VarFun2, x_value, dim=2, eps = 1e-5, percent = 1e-5).GetGrad(0.1,x_value)
#     CCentralDiff(Test2VarFun4, x_value, dim=2, eps = 1e-5, percent = 1e-2).GetGrad(0.1)
