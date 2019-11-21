import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import time

def TestLineFun1(x):
    return x**4-14*(x**3)+60*(x**2)-70*x
# x: 0.75 f(x): -24

def TestLineFun3(x):
    return -(108*x-x**3)/4
# x:6 f(x): -108

#Golden Section search

#phase one

def Phase_one(rate,update=1.618):
    
    for i in range(100):
        value = rate * (update)**(i-1) * (1 + update)
        minize_value = TestLineFun3(value)
        print(i,value,minize_value)
        #最後收斂極限
        if (minize_value >= TestLineFun3(0)):
            break
            
    return value
        

# phase two

def Eigenvalue(scaler,low_bond,up_bond):
    #計算upbond、lowbond 間距
    return scaler * (up_bond - low_bond)

def Iteration_define(x_1_minize_value,x_2_minize_value,x_1,x_2,up_side_value,low_side_value,scaler):
    
    if ( x_1_minize_value < x_2_minize_value):
        up_side_value = x_2
        x_2 = x_1
        x_1 = low_side_value +  Eigenvalue(scaler,low_side_value,up_side_value)

    if ( x_1_minize_value > x_2_minize_value):
        low_side_value = x_1
        x_1 = x_2
        x_2 = up_side_value - Eigenvalue(scaler,low_side_value,up_side_value)

    if ( x_1_minize_value == x_2_minize_value):
        low_side_value = x_1
        up_side_value = x_2

    return x_1_minize_value,x_2_minize_value,x_1,x_2,up_side_value,low_side_value,scaler


def Phase_two(limit,low_bond=0,scaler=0.382):
    interval = 50
    print(interval)
    
    x_1 = low_bond + scaler * interval
    x_2 = low_bond + (1 - scaler) * interval
    up_bond = low_bond + interval
    #x_1、x_2間距 = (1 - 2*scaler) * eigenvalue

    for i in range(0,100):
        x_1_minize_value = TestLineFun1(x_1)
        x_2_minize_value = TestLineFun1(x_2)

        x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler =\
             Iteration_define(x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler)

        print(i,up_bond,low_bond,TestLineFun1((x_1+x_2)/2))
        #最後收斂極限
        if ( up_bond - low_bond < limit):
            finel_value = (x_2 + x_1) / 2
            final_minize_value = TestLineFun1(finel_value)
            print(i,finel_value,final_minize_value)
            break
        

# Fibonacci Search Algorithm Phase 2   特色:scaler會隨著迭代變動
fibonacci_count = []


def fibon(n):
    # fibonacci陣列
    a = b = 1
    for i in range(n):
        yield a
        a, b = b, a + b
for x in fibon(20):
    fibonacci_count.append(x)


def Iteration_Fib_define(x_1_minize_value,x_2_minize_value,x_1,x_2,up_side_value,low_side_value,scaler,time):
    
    if ( x_1_minize_value < x_2_minize_value):
        up_side_value = x_2
        x_2 = x_1
        scaler = 1 - (fibonacci_count[time-1]/fibonacci_count[time]) #計算下一個scaler
        eigenvalue = Eigenvalue(scaler,low_side_value,up_side_value)
        x_1 = low_side_value + eigenvalue

    if ( x_1_minize_value > x_2_minize_value):
        low_side_value = x_1
        x_1 = x_2
        scaler = 1 - (fibonacci_count[time-1]/fibonacci_count[time]) #計算下一個scaler
        eigenvalue = Eigenvalue(scaler,low_side_value,up_side_value)
        x_2 = up_side_value - eigenvalue

    if ( x_1_minize_value == x_2_minize_value):
        low_side_value = x_1
        up_side_value = x_2
        scaler = 1 - (fibonacci_count[time-1]/fibonacci_count[time]) #計算下一個scaler


    return x_1_minize_value,x_2_minize_value,x_1,x_2,up_side_value,low_side_value,scaler

def Final_Fib_iteration(x_1,x_2,low_bond,up_bond,final_range):
    x_1_minize_value = TestLineFun1(x_1)
    x_2_minize_value = TestLineFun1(x_2)

    if ( x_1_minize_value < x_2_minize_value):
        up_bond = x_2
        low_bond = low_bond
        eigen = up_bond - low_bond
        if (eigen > final_range):
            print('weird situation: eigen > limit')
        else:
            func_value = 0.5 * (up_bond + low_bond)
            print('result:',TestLineFun1(func_value))

    if ( x_1_minize_value > x_2_minize_value):
        up_bond = x_1
        low_bond = up_bond
        eigen = up_bond - low_bond
        if (eigen > final_range):
            print('weird situation: eigen > limit')
        else:
            func_value = 0.5 * (up_bond + low_bond)
            print('result:',TestLineFun1(func_value))

    if ( x_1_minize_value == x_2_minize_value):
        up_bond = x_1
        low_bond = x_2
        eigen = abs(up_bond - low_bond)
        if (eigen > final_range):
            print('weird situation: eigen > limit')
        else:
            func_value = 0.5 * (up_bond + low_bond)
            print('result:',TestLineFun1(func_value))

def Fibonacci_Search(limit,final_range,low_bond,up_bond):
    #計算迭代次數:(1+2*limit)/fibonacci_(n+1) <= final_uncertain_range/initial_uncertain_range
    F_N = ((1 + 2*limit) * (up_bond - low_bond)) / final_range   # F_N = fibonacci 某一值
    if F_N not in fibonacci_count:        
        N = sorted(fibonacci_count + [F_N]).index(F_N) + 1 - 1  # +1為選取我要的fibonacci數列中的值 N = 迭代次數 (F_N 為 N+1 所以算出來要減1)
    else:
        N = fibonacci_count.index(F_N) - 1

    for i in range(N,0,-1):
        #根據迭代次數做運算
        if i == 1:
            #最後迭代
            scaler = 0.5 - limit
            eigenvalue = Eigenvalue(scaler,low_bond,up_bond)
            x_1 = low_bond + eigenvalue
            x_2 = up_bond - eigenvalue

            Final_Fib_iteration(x_1,x_2,low_bond,up_bond,final_range)

            
        else:
            if i == N:
                scaler = 1 - (fibonacci_count[i]/fibonacci_count[i+1])
                eigenvalue = Eigenvalue(scaler,low_bond,up_bond)
                
                #low、up_bond 分別為上下邊界，x1、x2為產生的點
                x_1 = low_bond + eigenvalue
                x_2 = up_bond - eigenvalue
                x_1_minize_value = TestLineFun1(x_1)
                x_2_minize_value = TestLineFun1(x_2)

                x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler =\
                    Iteration_Fib_define(x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler,i)
                
            else:
                x_1_minize_value = TestLineFun1(x_1)
                x_2_minize_value = TestLineFun1(x_2)

                x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler =\
                    Iteration_Fib_define(x_1_minize_value,x_2_minize_value,x_1,x_2,up_bond,low_bond,scaler,i)

    


if __name__ == '__main__':
    #phase 1
    step_size = 0.01
    update_parameter = 1.618
    Phase_one(step_size,update_parameter)

    #phase 2
    scaler = 0.382
    limitation = 0.001
    lower_bond = 1 
    upper_bond = 50   
    
    #Phase_two(limitation)

    #Fibonacci Search Algorithm Phase 2
    limitation = 0.1
    final_range = 0.3
    lower_bond = 1 
    upper_bond = 51 
    #Fibonacci_Search(limitation,final_range,lower_bond,upper_bond)
