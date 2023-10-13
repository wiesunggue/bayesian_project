import pickle
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.stats import gamma
from scipy.optimize import bisect,fsolve
from scipy.integrate import quad,nquad,dblquad,tplquad
from hidden_function import P_A,P_B,P_C,P_d1,P_d2
from numpy.random import uniform,gamma,normal,exponential
x = list(map(Decimal,np.linspace(0.001,100,1000)))

aa,bb,cc,d1,d2=0.2,0.3,0.4,0.1,1
aa, bb, cc, d1, d2 = [Decimal(i) for i in (aa,bb,cc,d1,d2)]  # decimal 로 형변환
# gamma function에 들어가는 상수 alpha beta


sample_size = 100
M = lambda aa,bb,cc,d1,d2:(Decimal(quad(lambda y : (-bb*Decimal(y)).exp()/(aa+cc*Decimal(y)),d1,d2)[0]),0)
mconst = (M(aa,bb,cc,d1,d2)[0]) # aa=2,bb=3,cc=4,d1=1,d2=10에서 0.002323084 R코드와 비교해서 잘 나오는지 확인 완료
# 내부 함수
# pdf함수 정의, x=0.1,aa=2,bb=3,cc=4,d1=1,d2=10조건에서 3.459362 확인 완료, 적분도 결과 확인 완료
dhtexp = lambda x,aa,bb,cc,d1,d2:(-aa*Decimal(x)).exp()*((-(bb+cc*Decimal(x))*d1).exp()-(-(bb+cc*Decimal(x))*d2).exp())/(mconst*(bb+cc*Decimal(x)))
# cdf함수 정의
phtexp = lambda x,aa,bb,cc,d1,d2:quad(dhtexp,0,x,args=(aa,bb,cc,d1,d2))
# cdf에서 역함수를 구하는 대신 qq에 해당하는 값을 구하는 함수 qq=0.4에서 0.07197확인 완료
# cdf이므로 근을 찾기 위해 좀더 효율적인 이분탐색 진행하였음
qhtexp = lambda qq,aa,bb,cc,d1,d2,imax=599:bisect(lambda x:(phtexp(x,aa,bb,cc,d1,d2)[0]-qq),0,imax)
#qhtexp = lambda qq,aa,bb,cc,d1,d2,imax=599:bisect(lambda x:(phtexp(x,aa,bb,cc,d1,d2)[0]-qq),0,imax)
# 0~1의 균일 분포에서 뽑은 랜덤한 값을 qhtexp에 넣은 값의 배열을 추출하는 과정
rhtexp = lambda N,aa,bb,cc,d1,d2,imax=599:([Decimal(qhtexp(uniform(),aa,bb,cc,d1,d2)) for _ in range(N)])

# data 500개면 약 데이터의 총 합이 70정도됨

var = rhtexp(sample_size,aa,bb,cc,d1,d2)
mode = 2
if mode==1:
    y = np.array([P_A(i,bb,cc,d1,d2,var) for i in x])
    plt.plot(x,y)
    p1 = np.argmax(y)
    print(x[p1])
    plt.scatter(x[p1],max(y),label="{:.4f} argmax(y), max y value {} ".format(x[p1],max(y)))
    plt.legend()
    plt.savefig('full conditional A')
elif mode==2:
    y = np.array([P_B(aa,i,cc,d1,d2,var) for i in x])
    plt.plot(x,y)
    p1 = np.argmax(y)
    plt.scatter(x[p1],max(y),label="{:.4f} argmax(y), max y value {} ".format(x[p1],max(y)))
    plt.legend()
    plt.savefig('full conditional B2')
elif mode == 3:
    y = np.array([P_C(aa, bb, i, d1, d2, var) for i in x])
    plt.plot(x, y)
    p1 = np.argmax(y)
    plt.scatter(x[p1], max(y), label="{:.4f} argmax(y), max y value {} ".format(x[p1], max(y)))
    plt.legend()
    plt.savefig('full conditional C')
elif mode == 4:
    y = np.array([P_d1(aa, bb, cc, i, d2, var) for i in x])
    plt.plot(x, y)
    p1 = np.argmax(y)
    plt.scatter(x[p1], max(y), label="{:.4f} argmax(y), max y value {} ".format(x[p1], max(y)))
    plt.legend()
    plt.savefig('full conditional d1')
elif mode == 5:
    y = np.array([P_d2(aa, bb, cc, d1, i, var) for i in x])
    plt.plot(x, y)
    p1 = np.argmax(y)
    plt.scatter(x[p1], max(y), label="{:.4f} argmax(y), max y value {} ".format(x[p1], max(y)))
    plt.legend()
    plt.savefig('full conditional d2')