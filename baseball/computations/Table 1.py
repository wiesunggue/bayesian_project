#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 라이브러리 불러오기
import numpy as np
import pandas as pd
import math
from math import e
ln = np.log  # assign the numpy log function to a new function called ln
from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')

import sys
import time
import datetime


# In[2]:


df = pd.read_csv("MLB datasets-2021-demeaned.csv")
L = 60; Time = 6
m = df[["m"]].values.flatten() # 순서대로 m을 가짐(선수 상관 x)
x1 = df[["x_1"]].values.flatten() #순서대로 x1을 가짐(선수 상관 x)
x2 = df[["x_2"]].values.flatten() #순서대로 x2를 가짐(선수 상관 x)

Z_demd = df.drop(['Team','Player','month','m','x_1','x_2'], axis=1)
Z_demd.head()


# ---

# In[3]:


z_p = Z_demd.loc[:,["Intercept","WPA", "Cent%",
                           "BABIP","BB/K", "LD%", "GB%", "Oppo%"]]
z_q = Z_demd.loc[:,["Intercept","WPA", "Cent%",
                           "FB%", "HR/FB", "Pull%"]]

####################################################
B_p = [-2.05,0.06,-0.64,3.37,0.14,-0.2,-0.16,0.50] 
B_q = [-0.5,0.72,-0.64,0.60,0.90,1.83] 
params = B_p + B_q
####################################################
d = len(params)

phi=100


# ---

# In[5]:


def fp(tta,bps,phi): # 함수의 목적은?
    bps[k] = tta
    Z_Bp = z_p.dot(np.diag(bps))
    logit_p = np.array(Z_Bp.sum(axis=1))
    ests = (tta * (z_p.iloc[:, [k]].to_numpy().flatten()) * x1 - m*ln(1+np.exp(logit_p))).sum()-(tta**2)/(2*(phi**2))
    return ests

def fq(tta,bqs,phi): # 함수의 목적은?
    bqs[k] = tta
    Z_Bq = z_q.dot(np.diag(bqs))
    logit_q = np.array(Z_Bq.sum(axis=1))
    ests = (tta * (z_q.iloc[:, [k]].to_numpy().flatten()) * x2 - x1*ln(1+np.exp(logit_q))).sum()-(tta**2)/(2*(phi**2))
    return(ests)


# In[6]:


d = len(params)
s = 2.4/math.sqrt(d) # 아마 자유도로 나눠서 무슨 계수 구하는거인듯

np.random.seed(10)
eps = np.random.uniform(0,1,size=d)*0.05 # 0~1사이, 파라메터 개수 만큼 균등분포에서 랜덤값을 추출
Var_0  = np.random.uniform(5,20,size=d) # 5~20 사이의 랜덤값 추출
def EmpVar(nth):
    if i == 0:
        rslt = Var_0[nth]
    elif i == 1:
        rslt = s*eps[nth]
    else: 
        rslt = s*np.var(u.iloc[0:(i-1),nth]) + s*eps[nth]
    return(rslt)


# ---

# In[7]:


itr = 6000
burn = 1000
thin = 1


# ---

# In[8]:


u_colnames = ["bp" + str(num1) for num1, in zip(range(len(B_p)))]+[
    "bq" + str(num1) for num1, in zip(range(len(B_q)))] # 변수명 재정의(논문식 표기를 위함)

u = np.zeros((itr+1, d)) # 6001*14의 행렬 선언(계산 결과 기록용)
u = pd.DataFrame(u, columns = u_colnames) #판다스로 형변환 + column 이름 붙이기
u.iloc[0,:] = params # 첫 시행은 랜덤 추출된 숫자들을 가짐
u.head()


# ---

# # Start the iteration

# In[9]:


t1 = time.time()
######################################################################################
for i in range(itr):
    np.random.seed(i)
    ############## beta_pk 추정 ##############
    for k in range(0, len(B_p)): # 하나씩 업데이트
        u_curr = u.iloc[i,k]  # 이전 상태 불러오기
        u_cand = float(np.random.normal(u_curr,np.sqrt(EmpVar(k)),size=1)) # 정규 분포에서 추출

        B_p2 = B_p      
        B_p2[k] = u_cand

        r = np.exp(fp(u_cand,B_p2,phi) - fp(u_curr,B_p,phi)) #업데이트 비율
        values = np.array([u_cand,u_curr])
        p = np.array([r, 1-r])
        if r >= 1:
            B_p[k] = u_cand
        else:
            B_p[k] = float(np.random.choice(values, size=1, p=p))

        u.iloc[i+1,k] = B_p[k]
  
    ############## beta_qk 추정 ##############
    for k in range(0, len(B_q)):
        nth = len(B_p)+k
        u_curr = u.iloc[i,nth]  
        u_cand = float(np.random.normal(u_curr,np.sqrt(EmpVar(nth)),size=1))

        B_q2 = B_q      
        B_q2[k] = u_cand
        
        r = np.exp(fq(u_cand,B_q2,phi) - fq(u_curr,B_q,phi))
        values = np.array([u_cand,u_curr])
        p = np.array([r, 1-r])
        if r >= 1:
            B_q[k] = u_cand
        else:
            B_q[k] = float(np.random.choice(values, size=1, p=p))

        u.iloc[i+1,nth] = B_q[k]
        
    sys.stdout.write('\rIteration is %d' %i)  
    sys.stdout.flush()
######################################################################################
u1 = u.drop(labels=range(0,burn+1),axis=0).reset_index(drop = True) 

t2 = time.time()
sec = t2-t1

result_list = str(datetime.timedelta(seconds=sec)).split(".")
print("\nTime elapsed:",result_list[0])


# ---

# # Results

# In[11]:


z = norm.ppf(.975)  ## Percent point function

stats_u = u1.describe().transpose()[['mean','std']]

B_Mean = stats_u.loc[:,['mean']].values.flatten()
B_SD = stats_u.loc[:,['std']].values.flatten()

B_L = B_Mean - z * B_SD
B_U = B_Mean + z * B_SD


# In[12]:


TABLE = pd.DataFrame({"Mean":B_Mean, "SD":B_SD, "Lower": B_L, "Upper": B_U}, index = stats_u.index)

TABLE

print(TABLE)
# ---

# In[13]:


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform


# In[14]:


## 그래프에서 마이너스 기호가 표시되도록 하는 설정
matplotlib.rcParams['axes.unicode_minus'] = False

## Add space between the ticklabels and the axes in matplotlib.
matplotlib.rcParams['xtick.major.pad']='10'
matplotlib.rcParams['ytick.major.pad']='6'


# In[17]:


df = u1

fig, axes = plt.subplots(4, 4, figsize=(13,8))
fig.suptitle('Traceplots of parameters (Model0)', fontsize=20, position=(0.5, 1),fontweight="bold")
# fig.subplots_adjust(hspace=1,wspace=0.5)

for col, ax in zip(df.columns, axes.flatten()):
    ax.plot(df.index, df[col])
    
    ax.set_xticks(np.linspace(start=0, stop=df.shape[0], num=6))
    
    ax.set_ylabel(r"$\beta$"+col[1:3], fontsize=12, rotation="horizontal",labelpad=10)
#     show_y = np.linspace(start=df[col].min(), stop=df[col].max(), num=4)
    show_y = np.linspace(start=df.min(axis=0)[col], stop=df.max(axis=0)[col], num=4)
    ax.set_yticks(np.round(show_y,2))
    ax.tick_params(axis='y', labelsize=10)
    
fig.delaxes(axes[3][2]);fig.delaxes(axes[3][3])
plt.tight_layout()
plt.show()

