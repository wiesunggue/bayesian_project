import time
from math import exp,log10,log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sy
import math
import scipy.integrate as integrate
import scipy as sc
from numpy import sqrt, exp

start=time.time()
import scipy as sc
from scipy.optimize import bisect,fsolve
from scipy.integrate import quad,nquad,dblquad,tplquad
end=time.time()
print("scipy : 모듈 불러오는 시간: ",end-start)
from decimal import Decimal
aa,bb,cc,d1,d2=2,3,4,1,10
# gamma function에 들어가는 상수 alpha beta
a1,b1=1,1
a2,b2=1,1
a3,b3=1,1

def c_func(ld):
    g_x=[]
    one=Decimal(1)
    two=Decimal(2)
    four = Decimal(4)
    sqr = Decimal(sqrt(2))
    ld0=5
    for x in range(1,16):
        g_x.append(Decimal(((4*ld0*ld)**x)/(math.factorial(2*x)*((2**x)+sqrt(2)))))
        #     g.x[x] <- ((4*ld.0*ld)^x)/((factorial(2*x))*((2^x)+sqrt(2)))

    #compute = np.cosh(2*sqrt(ld0*ld))
    ld0 = Decimal(ld0)
    ld = Decimal(ld)
    compute = (exp(two*sqrt(ld0*ld))+exp(-two*sqrt(ld0*ld)))/two
    #print(compute,ld)
    #print(sum(g_x))
    return ((exp(-ld)*(-sqr*Decimal(sum(g_x))+Decimal(compute)-one))/(sqrt(ld)*(one-exp(-ld))))
#     ((exp(-ld)*(-sqrt(2)*sum(g.x) + cosh(2*sqrt(ld.0*ld))-1)) / (sqrt(ld)*(1-exp(-ld)))) ## intrinsic prior of m2
#print('사용 시간',end-time)


M = lambda aa,bb,cc,d1,d2:quad(lambda y:exp(-bb*y)/(aa+cc*y),d1,d2)
mconst = M(aa,bb,cc,d1,d2)[0] # aa=2,bb=3,cc=4,d1=1,d2=10에서 0.002323084 R코드와 비교해서 잘 나오는지 확인 완료
# Arnold Strauss bivariate ex-ponential distribution
AS_bi_exp_dist = lambda x,y : exp(-(a*x+b*y+c*x*y))/mconst
# pdf함수 정의, x=0.1,aa=2,bb=3,cc=4,d1=1,d2=10조건에서 3.459362 확인 완료, 적분도 결과 확인 완료
dhtexp = lambda x,aa,bb,cc,d1,d2:exp(-aa*x)*(exp(-(bb+cc*x)*d1)-exp(-(bb+cc*x)*d2))/(mconst*(bb+cc*x))

# cdf함수 정의
phtexp = lambda x,aa,bb,cc,d1,d2:quad(dhtexp,0,x,args=(aa,bb,cc,d1,d2))

# cdf에서 역함수를 구하는 대신 qq에 해당하는 값을 구하는 함수 qq=0.4에서 0.07197확인 완료
# cdf이므로 근을 찾기 위해 좀더 효율적인 이분탐색 진행하였음
qhtexp = lambda qq,aa,bb,cc,d1,d2,imax=599:bisect(lambda x:(phtexp(x,aa,bb,cc,d1,d2)[0]-qq),0,imax)

# 0~1의 균일 분포에서 뽑은 랜덤한 값을 qhtexp에 넣은 값의 배열을 추출하는 과정
rhtexp = lambda N,aa,bb,cc,d1,d2,imax=599:np.array([qhtexp(np.random.rand(1),aa,bb,cc,d1,d2) for _ in range(N)])

#print(1/8)
#print(sc.integrate.nquad(lambda x,y:f(x,y)*g(y),[[0,1],[0,1]]))
#print(sy.integrate(sy.exp(-b*y1)/a+c*y1,y1))
M = lambda aa,bb,cc,d1,d2:quad(lambda y:exp(-bb*y)/(aa+cc*y),d1,d2)

def bounds_y():
    return [0, 100]

def bounds_x(d1):
    return [d1+1, 100]
sample_size = 1
var = rhtexp(sample_size,aa,bb,cc,d1,d2)
start = time.time()
#print(nquad(lambda aa,bb,cc,y,d2,d1 : exp(-aa*len(var)*log(exp(-bb*y)/(aa+cc*y))-aa*(b1-a1+1+sum(var))),[[0,100],[0,100],[0,100],lambda d1,d2:[d1,d2],lambda d2:[0,d2],[0,10]]))
end = time.time()
print("실행시간 : ",end-start)

aa = np.random.uniform(size=100000)
bb = np.sqrt(1/aa)-1
bb = sorted(bb)
print(bb[-1],bb[0])
print(np.mean(bb),np.var(bb))
plt.hist(bb,bins=1000)
plt.show()