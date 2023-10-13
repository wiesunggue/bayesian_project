import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 불러올 파일 이름을 입력
get_data = pd.read_csv("20save_data(0.3).csv")
sim_data=get_data.dropna(axis=0) # 결측치 제거
result1=np.zeros(len(sim_data)) # 결과를 저장할 배열
result2=np.zeros(len(sim_data))
for i in range(len(sim_data)):
    a,b,c,d = sim_data.iloc[i,1:5]
    result1[i] = b/a*c # BF*CF
    result2[i] = b/a*d
    
print("AIBF 방법")
print('1보다 큰 비율 : ',sum([1 if i>1 else 0 for i in result1])/len(result1))
print('평균 : ',sum(result1)/len(result1))
print('적분 실패 : ',1000-len(sim_data))
print("fractional 방법")
print('성공 횟수 : ',sum([1 if i>1 else 0 for i in result2])/len(result2))
print('1보다 큰 비율 : ',sum(result2)/len(result2))
print('적분 실패 : ',len(get_data)-len(sim_data))

plt.plot(sim_data.iloc[:,1],label='m1')
plt.plot(sim_data.iloc[:,2],label='m2')
plt.plot(sim_data.iloc[:,3],label='AIBF')
plt.plot(sim_data.iloc[:,4],label='fractional')
plt.legend()
plt.show()