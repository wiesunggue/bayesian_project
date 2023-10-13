import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sy
import math
import scipy.integrate as integrate
import scipy as sc
from numpy import sqrt, exp

from decimal import Decimal
import time
####
#### omega
ld = 3
Trom=0.3
om=0.5
N=50 # 생성할 데이터셋의 i수
inf2=500
iter=100
set_seed=41

# 분산 최적화

def ZIP(ld,size=N):
    '''zip 값을 랜덤하게 추출하는 함수'''
    arr=[]
    for i in range(size):
        global set_seed
        set_seed=set_seed+1
        np.random.seed(set_seed)
        arr.append(*np.random.choice((0,*np.random.poisson(lam=ld, size=1)),size=1,p=[Trom,1-Trom]))

    return arr


def inner_F(x):
    out = 6*(1-om)*(om+(1-om)*2**(-x-0.5))/1+2**(-x+0.5)

    return out


def C_func(ld=ld):
    '''intrinsic prior을 변수에 대해 적분하는 부분'''
    g_x=0
    global om
    om = Decimal(om)
    ld = Decimal(ld)
    one = Decimal(1)
    two = Decimal(2)
    sqr = Decimal(sqrt(2))
    for x in range(1,16):
        g_x = g_x+((ld**x)*(om+(1-om)/(sqr*2**x)))/(math.factorial(x)*(2**x+sqr))
    g_x = Decimal(g_x)
    return (exp(-ld)*(-sqr*g_x+om*(exp(ld)-one)+(one-om)*(exp(ld/two)-one)/sqr)/(sqrt(ld)*(one-exp(-ld))))


def C_float(ld=ld):
    '''intrinsic prior을 변수에 대해 적분하는 부분'''
    g_x=[]
    om=0.5
    for x in range(1,16):
        g_x.append(((ld**x)*(om+(1-om)/(sqrt(2)*2**x)))/(math.factorial(x)*(2**x+sqrt(2))))

    return ((exp(-ld)*(-sqrt(2)*sum(g_x)+om*(exp(ld)-1)+(1-om)*(exp(ld/2)-1)/sqrt(2)))/(sqrt(ld)*(1-exp(-ld))))


# 적분할 범위에 대한 정ㄹ보 1부터 1000000까지 1000씩 증가시키면서 데이터 생성
dist = list(range(1,2000000,100000))

ans=[]
start = time.time()
for i in dist:
    ans.append(integrate.quad(C_func,0,i,points=[0,1000]))
print('0부터 2300000까지 적분한 값 : ',integrate.quad(C_func,0,2000000,points=[0,1000]))
print('0부터 500    까지 적분한 값 : ',integrate.quad(C_func,0,1000000,points=[0,100]))
print('0부터 700    까지 적분한 값 : ',integrate.quad(C_func,0,500000,points=[0,100]))
end = time.time()
print('걸린 시간',end-start)
x = sy.symbols()
plt.plot(dist, ans)
plt.xlabel('로그 스케일에서 함수 값')

plt.show()
print(C_float(1000))



#print(integrate.quad(f,0,10))

print(exp(Decimal(2300000)))