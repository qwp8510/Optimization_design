import time
import numpy as np

#LineSearch
def TestLineFun1(x):
    return x**4-14*(x**3)+60*(x**2)-70*x
# x: 0.75 f(x): -24
def TestLineFun2(x):
    return (0.65-0.75/(1+x**2))-0.65*x*np.arctan2(1,x)
# x: 0.48 f(x): -0.3
def TestLineFun3(x):
    return -(108*x-x**3)/4
# x:6 f(x): -108
#Golden Section search
#phase one
class CGSSearch():
    def __init__(self,costfun,x=0,d=1,eps=0.01):
        self.costfun = costfun
        self.x = x
        self.d = d
        self.eps = eps

    def set_costfun(self,costfun):
        return costfun(self.x)

    def set_x(self,x):
        # 將x傳回function
        self.x = x
        return self.set_costfun(self.costfun)

    def set_d(self,d):
        self.d = d

    #def Runsearch(self,rate,update):    
    def Runsearch(self,step_size):
        return self.__Phase_two()

    def __Phase_one(self,step_size,update=1.618):
        i = 34
        value = step_size * (update)**(i-1) * (1 + update)
        return value

    # phase two

    def Eigenvalue(self,scaler,low_bond,up_bond):
        #計算upbond、lowbond 間距
        return scaler * (up_bond - low_bond)

    def Iteration_define(self,x_1_minize_value,x_2_minize_value,x_1,x_2,up_side_value,low_side_value,scaler):
        
        if ( x_1_minize_value < x_2_minize_value):
            up_side_value = x_2
            x_2 = x_1
            x_1 = low_side_value + self.Eigenvalue(scaler,low_side_value,up_side_value)

        if ( x_1_minize_value > x_2_minize_value):
            low_side_value = x_1
            x_1 = x_2
            x_2 = up_side_value - self.Eigenvalue(scaler,low_side_value,up_side_value)

        if ( x_1_minize_value == x_2_minize_value):
            low_side_value = x_1
            up_side_value = x_2

        return x_1_minize_value,x_2_minize_value,x_1,x_2,up_side_value,low_side_value,scaler


    def __Phase_two(self,low_bond=0,scaler=0.382):
        interval = self.__Phase_one(step_size=0.001)
        
        x_1 = low_bond + scaler * interval
        x_2 = low_bond + (1 - scaler) * interval
        up_bond = low_bond + interval   
        #x_1、x_2間距 = (1 - 2*scaler) * eigenvalue

        for i in range(0,1000):
            x_1_minize_value = self.set_x(x_1)
            x_2_minize_value = self.set_x(x_2)

            x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler =\
                self.Iteration_define(x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler)
            print('pyline search 收斂過程',(x_2 + x_1)/2,self.set_x((x_2 + x_1)/2))
            #最後收斂極限
            if ( up_bond - low_bond < self.eps):
                finel_value = (up_bond + low_bond) * 0.5
                final_minize_value = self.set_x(finel_value)
                print('pyline search 收斂結果',i,finel_value,final_minize_value)
                return finel_value
                break
        

#Fibonacci Search Algorithm Phase 2   特色:scaler會隨著迭代變動

class CFiSearch():
    fibonacci_count = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 
                        17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 
                        5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141, 267914296, 
                        433494437, 701408733, 1134903170, 1836311903, 2971215073, 4807526976, 7778742049, 12586269025 ]

    def __init__(self,costfun,x=0,d=1,eps=0.01):
        self.costfun = costfun
        self.x = x
        self.d = d
        self.eps = eps

    def set_costfun(self,costfun):
        return costfun(self.x)

    def set_x(self,x):
        # 將x傳回function
        self.x = x
        return self.set_costfun(self.costfun)

    def set_d(self,d):
        self.d = d

    def Runsearch(self,eps,final_range,low_bond,up_bond):
        return self.__Fibonacci_Search(self.eps,final_range,low_bond,up_bond)


    def Eigenvalue(self,scaler,low_bond,up_bond):
        #計算upbond、lowbond 間距
        return scaler * (up_bond - low_bond)

    def Iteration_Fib_define(self,x_1_minize_value,x_2_minize_value,x_1,x_2,up_side_value,low_side_value,scaler,time):
        
        if ( x_1_minize_value < x_2_minize_value):
            up_side_value = x_2
            x_2 = x_1
            scaler = 1 - (self.fibonacci_count[time-1]/self.fibonacci_count[time]) #計算下一個scaler
            eigenvalue = self.Eigenvalue(scaler,low_side_value,up_side_value)
            x_1 = low_side_value + eigenvalue

        if ( x_1_minize_value > x_2_minize_value):
            low_side_value = x_1
            x_1 = x_2
            scaler = 1 - (self.fibonacci_count[time-1]/self.fibonacci_count[time]) #計算下一個scaler
            eigenvalue = self.Eigenvalue(scaler,low_side_value,up_side_value)
            x_2 = up_side_value - eigenvalue

        if ( x_1_minize_value == x_2_minize_value):
            low_side_value = x_1
            up_side_value = x_2
            scaler = 1 - (self.fibonacci_count[time-1]/self.fibonacci_count[time]) #計算下一個scaler


        return x_1_minize_value,x_2_minize_value,x_1,x_2,up_side_value,low_side_value,scaler

    def Final_Fib_iteration(self,x_1,x_2,low_bond,up_bond,final_range):
        x_1_minize_value = self.set_x(x_1)
        x_2_minize_value = self.set_x(x_2)

        if ( x_1_minize_value < x_2_minize_value):
            up_bond = x_2
            low_bond = low_bond
            eigen = up_bond - low_bond
            if (eigen > final_range):
                print('weird situation: eigen > limit')
            else:
                func_value = 0.5 * (up_bond + low_bond)
                print(func_value,'result:',self.set_x(func_value))

        if ( x_1_minize_value > x_2_minize_value):
            up_bond = x_1
            low_bond = up_bond
            eigen = up_bond - low_bond
            if (eigen > final_range):
                print('weird situation: eigen > limit')
            else:
                func_value = 0.5 * (up_bond + low_bond)
                print(func_value,'result:',self.set_x(func_value))

        if ( x_1_minize_value == x_2_minize_value):
            up_bond = x_1
            low_bond = x_2
            eigen = abs(up_bond - low_bond)
            if (eigen > final_range):
                print('weird situation: eigen > limit')
            else:
                func_value = 0.5 * (up_bond + low_bond)
                print(func_value,'result:',self.set_x(func_value))

    def __Fibonacci_Search(self,limit,final_range,low_bond,up_bond):
        # self.fibonacci_count = [self.fibon(n) for n in range(20)]
        #計算迭代次數:(1+2*limit)/fibonacci_(n+1) <= final_uncertain_range/initial_uncertain_range
        F_N = ((1 + 2*limit) * (up_bond - low_bond)) / final_range   # F_N = fibonacci 某一值
        if F_N not in self.fibonacci_count:        
            N = sorted(self.fibonacci_count + [F_N]).index(F_N) + 1 - 1  # +1為選取我要的fibonacci數列中的值 N = 迭代次數 (F_N 為 N+1 所以算出來要減1)
        else:
            N = self.fibonacci_count.index(F_N) - 1

        for i in range(N,0,-1):
            #根據迭代次數做運算
            if i == 1:
                #最後迭代
                scaler = 0.5 - limit
                eigenvalue = self.Eigenvalue(scaler,low_bond,up_bond)
                x_1 = low_bond + eigenvalue
                x_2 = up_bond - eigenvalue

                self.Final_Fib_iteration(x_1,x_2,low_bond,up_bond,final_range)
                
            else:
                if i == N:
                    #初次迭代
                    scaler = 1 - (self.fibonacci_count[i]/self.fibonacci_count[i+1])
                    eigenvalue = self.Eigenvalue(scaler,low_bond,up_bond)
                    
                    #low、up_bond 分別為上下邊界，x1、x2為產生的點
                    x_1 = low_bond + eigenvalue
                    x_2 = up_bond - eigenvalue
                    x_1_minize_value = self.set_x(x_1)
                    x_2_minize_value = self.set_x(x_2)

                    x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler =\
                        self.Iteration_Fib_define(x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler,i)
                    
                else:
                    #大部分迭代
                    x_1_minize_value = self.set_x(x_1)
                    x_2_minize_value = self.set_x(x_2)

                    x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler =\
                        self.Iteration_Fib_define(x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler,i)


if __name__ == '__main__':

    #phase 2
    
    CGSSearch(TestLineFun1).Runsearch(0.01)

    #Fibonacci Search Algorithm Phase 2
    limitation = 0.001
    final_range = 0.3
    lower_bond = 0 
    upper_bond = 51

    #CFiSearch(TestLineFun1,x=0,d=1,eps=0.001).Runsearch(limitation,final_range,lower_bond,upper_bond)
    """
    測試紀錄:
        10/24: lower_bond != 0 時無法收斂，把upper_bond收斂效果也越差
        速度測試:@lru_測試: test3:0.00274
                           test2:0.00288
                           test1:0.0051~0.0034
                yield測試: test3:0.00372~0.0025
                           test2:0.00574~0.00262
                           test1:0.00514~0.0267

    """
