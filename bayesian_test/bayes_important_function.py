# 기본 로직
# 1. 중요함수 I(x)를 따르는 숫자 x1,...,xi를 생성한다
# 2. wi = g(xi)/I(xi)로 두고 계산
# 3. E[f(y|x)]=sum(f(y|xi)wi)/sum(wi)로 수렴한다.
from math import factorial
from functools import partial
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

def simulator(i):
    data = np.random.uniform(0,1,i)
    important_data = beta.ppf(data,30,51)
    weight = beta.pdf(data,30,50)/important_data
    # I = important function
    ans1 = sum(data*weight)/sum(weight) # I로 beta(4,5)인 경우
    ans2 = sum(data*beta.pdf(data,30,50))/i # 기본 몬테 카를로 시뮬레이션
    w = beta.pdf(data,30,50)/data
    ans3 = sum(data*w)/sum(w) # I로 균등분포인 경우
    return ans1,ans2,ans3

iteration = 1000
sample = 10000
ans_set1 = np.array([0]*iteration,dtype=np.float64) # I=Beta(6,5)
ans_set2 = np.array([0]*iteration,dtype=np.float64) # Basic Monte Carlo
ans_set3 = np.array([0]*iteration,dtype=np.float64) # I=Uniform(0,1)
ans_set4 = np.array([0]*iteration,dtype=np.float64) # Weighted Bootstrap
ans_set5 = np.array([0]*iteration,dtype=np.float64) # Acceptance-Rejection

for idx,i in enumerate([sample]*iteration):
    if (idx+1)%100==0:
        print(f'iteration : {idx+1}/{iteration}')
    ans_set1[idx],ans_set2[idx],ans_set3[idx]=simulator(i)

print(f"calculate expectation of Beta(30,50) ans=0.5, iteration={iteration}, sample size={sample}")
print("\t\tI=beta(30,51)\t\t\t\tBasic\t\t\t\t\tI=uniform")
print("mean",np.mean(ans_set1),'\t',np.mean(ans_set2),'\t',np.mean(ans_set3))
print("min ",np.min(ans_set1),'\t',np.min(ans_set2),'\t',np.min(ans_set3))
print("std ",np.std(ans_set1),'\t',np.std(ans_set3),'\t',np.std(ans_set3))

random_data = np.random.uniform(0,1,10**5)
plt.plot(random_data,beta.pdf(random_data,30,50),'.',label='beta(30,50)')
plt.plot(random_data,beta.pdf(random_data,30,51),'.',label='beta(30,51)')
plt.plot(random_data,[1 for i in random_data],'.',label='uniform(0,1)')
plt.legend()
plt.show()