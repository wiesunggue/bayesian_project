import numpy as np
from math import factorial,sqrt
from scipy.integrate import quad,dblquad

def ZIP(w=0.5,l=4,N=30): # 데이터 랜덤 생성
    bernoulli = np.random.choice(2,N,p=[w,1-w])
    ans = bernoulli * np.random.poisson(l,N)
    return ans

def Likelihood(w,l,data,prior = lambda l : l**(-0.5)): # 가능도의 정의
    alpha = sum([1 if i==0 else 0 for i in data])
    n = len(data)
    return (w+(1-w)*np.exp(-l))**alpha*((1-w)*np.exp(-l))**(n-alpha)*l**(sum(data))*prior(l)
    
def correction_fractional(w,data):
    alpha = sum([1 if i==0 else 0 for i in data])
    n = len(data)
    L2 = lambda w,l : ((w+(1-w)*np.exp(-l))**alpha*((1-w)*np.exp(-l))**(n-alpha)*l**(sum(data)))**(2/n)*l**(-0.5)
    L1 = lambda l : L2(w,l)
    corr_m1 = quad(L1,0,np.inf)
    corr_m2 = dblquad(L2,0,np.inf,0,1)
    return corr_m2[0]/corr_m1[0]
def correction_AIBF(w,data): # correction Factor를 구하는 함수
    data.sort()
    cnt = 0
    zero = sum([1if i==0 else 0 for i in data])
    if zero==len(data):
        return 0
    for i in data[zero:]:
        cnt+= 6*(1-w)*(w+(1-w)*2**(-i-0.5))/(1+2**(-i+0.5))
        
    return cnt/(len(data)-zero)

def intrinsic_prior(w,l,s,w_default=0.5):
    temp = 0
    for x in range(1,s+1):
        temp+= l**x/factorial(x)*1/(2**x+sqrt(2))*(w_default+(1-w_default)/sqrt(2)*2**(-x))
    return np.exp(-l)/(sqrt(l)*(1-np.exp(-l)))*(-sqrt(2)*temp+w_default*(np.exp(l)-1)+(1-w_default)/sqrt(2)*(np.exp(l/2)-1))
