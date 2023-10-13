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


# ---

# ### Simulationa setting 

# In[2]:


# Rep = 300
# itr= 4000
# burn = 1000
# ######### 임시 #########
Rep = 80
itr= 2200
burn = 200
#######################
phi = 100


# In[3]:


L=30; Time = 6; lt = L*Time; 
sd_a_l = 1; sd_z = 0.3

m =  np.tile(50, reps = lt)
phi= 100

#########  True value 넣을 list 만들기 #########
B_p = [0 for i in range(3)] 
B_q = [0 for i in range(3)] 
B_s = [0]
params = B_p + B_q + B_s
##############################################
d = len(params)

z = norm.ppf(.975)  


# In[4]:


u_colnames = ["bp" + str(num1) for num1, in zip(range(len(B_p)))]+[
    "bq" + str(num1) for num1, in zip(range(len(B_q)))]+["bst"]
u = pd.DataFrame(np.zeros((itr+1, d)), columns = u_colnames)

######################################################################################################
PostMean =  pd.DataFrame(np.zeros((Rep, d)), columns=u.columns) ## Posterior Mean을 위한 matrix
Bias =  pd.DataFrame(np.zeros((Rep, d)), columns=u.columns) ## Bias를 위한 matrix
MSE =  pd.DataFrame(np.zeros((Rep, d)), columns=u.columns) ## MSE를 위한 matrix
T_or_F =  pd.DataFrame(np.zeros((Rep, d)), columns=u.columns) ## Coverage Probability를 위한 matrix
AW =  pd.DataFrame(np.zeros((Rep, d)), columns=u.columns) ## Average Width를 위한 matrix


# In[5]:


def fp(tta,bps,a_l,phi):
    bps[k] = tta
    Z_Bp = z_p.dot(np.diag(bps))
    logit_p = np.array(Z_Bp.assign(a_l=a_l).sum(axis=1))
    ests = (tta *(z_p.iloc[:, [k]].to_numpy().flatten())* x1 - m*ln(1+np.exp(logit_p))).sum()-(tta**2)/(2*(phi**2))
    return ests

def fq(tta,bqs,bst,a_l,phi):
    bqs[k] = tta
    Z_Bq = z_q.dot(np.diag(bqs))
    logit_q = np.array(Z_Bq.assign(addcol=bst*a_l).sum(axis=1))
    ests = (tta * (z_q.iloc[:, [k]].to_numpy().flatten()) * x2 - x1*ln(1+np.exp(logit_q))).sum()-(tta**2)/(2*(phi**2))
    return(ests)

def fst(tta,bqs,a_l,phi):
    Z_Bq = z_q.dot(np.diag(bqs))
    logit_q = np.array(Z_Bq.assign(addcol=tta*a_l).sum(axis=1))
    ests = (tta*a_l*x2 - x1*ln(1+np.exp(logit_q))).sum()-(tta**2)/(2*(phi**2))
    return(ests)

def fa_l(tta,bps,bqs,bst):
    now_l = list(range(Time*lth, Time*(lth+1)))
    
    Z_Bp_l = z_p.iloc[now_l,:].dot(np.diag(bps))
    logit_pp = np.array(Z_Bp_l.assign(addcol=tta).sum(axis=1))
   
    Z_Bq_l = z_q.iloc[now_l,:].dot(np.diag(bqs))
    logit_qq = np.array(Z_Bq_l.assign(addcol=bst*tta).sum(axis=1))
    
    ests = (tta*(x1[now_l]+bst*x2[now_l]) - m[now_l]*ln(1+np.exp(logit_pp))
            - x1[now_l]*ln(1+np.exp(logit_qq))).sum() 
    return(ests)
################################################################################################################
## 분산조정 함수
def EmpVar(nth):
    if i == 0:
        rslt = Var_0[nth]
    elif i == 1:
        rslt = s*eps[nth]
    else: 
        rslt = s*np.var(u.iloc[0:(i-1),nth]) + s*eps[nth]
    return(rslt)


# ---

# In[6]:


t1 = time.time()

for j in range(Rep):
    np.random.seed(j)
    
    ################# True value #################
    B_p = [0,-1,2] # 차례대로 B_p0,B_p1,B_p2,B_p3
    B_q = [0,1,-2] # 차례대로 B_q0,B_q1,B_q2,B_q3
    B_s = [1]
    params = B_p + B_q + B_s

    Z = pd.DataFrame({'z0': np.ones(lt),
                      'z1': np.random.normal(0,sd_z,size=lt),
                      'z2': np.random.normal(0,sd_z,size=lt),
                      'z3': np.random.normal(0,sd_z,size=lt)})
    z_p = Z.loc[:,["z0","z1", "z2"]]
    z_q = Z.loc[:,["z0","z1", "z3"]]
    
    a_l = np.random.normal(0,sd_a_l,size=L)
    a_l = np.repeat(a_l, Time, axis=0) ## lt개 생성

    Z_Bp = z_p.dot(np.diag(B_p))
    logit_p = np.array(Z_Bp.assign(a_l=a_l)).sum(axis=1) ## lt개 생성

    Z_Bq = z_q.dot(np.diag(B_q))
    logit_q = np.array(Z_Bq.assign(addcol=B_s*a_l)).sum(axis=1) ## lt개 생성

    p = np.exp(logit_p)/(1 + np.exp(logit_p))
    q = np.exp(logit_q)/(1 + np.exp(logit_q))
    
    ## 방 1
#     x1 = m*p; x2 = x1*q
    ## 방 2
    x1 = np.random.binomial(m, p, size = lt)  
    x2 = np.random.binomial(x1, q, size = lt)


    ##############################################################################
    s = 2.4/math.sqrt(d-2)  
    eps = np.random.uniform(0,1,size=d)*0.05
    Var_0  = np.random.uniform(5,20,size=d)
 
    u = pd.DataFrame(np.zeros((itr+1, d)), columns = u_colnames)
    u.iloc[0,:] = params ## 초기값으로서 True값 대입
    
    for i in range(itr):
        ############## beta_pk 추정 ##############
        for k in range(1, len(B_p)):
            u_curr = u.iloc[i,k] ## current value
            u_cand = float(np.random.normal(u_curr,np.sqrt(EmpVar(k)),size=1))

            B_p2 = B_p     
            B_p2[k] = u_cand

            r = np.exp(fp(u_cand,B_p2,a_l,phi) - fp(u_curr,B_p,a_l,phi))
            values = np.array([u_cand,u_curr])
            p = np.array([r, 1-r])
            if r >= 1:
                B_p[k] = u_cand
            else:
                B_p[k] = float(np.random.choice(values, size=1, p=p))

            u.iloc[i+1,k] = B_p[k]
        ############## beta_qk 추정 ##############
        for k in range(1, len(B_q)):
            nth = len(B_p)+k
            u_curr = u.iloc[i,nth]  
            u_cand = float(np.random.normal(u_curr,np.sqrt(EmpVar(nth)),size=1))

            B_q2 = B_q     
            B_q2[k] = u_cand

            r = np.exp(fq(u_cand,B_q2,B_s,a_l,phi) - fq(u_curr,B_q,B_s,a_l,phi))
            values = np.array([u_cand,u_curr])
            p = np.array([r, 1-r])
            if r >= 1:
                B_q[k] = u_cand
            else:
                B_q[k] = float(np.random.choice(values, size=1, p=p))

            u.iloc[i+1,nth] = B_q[k]
        ############## beta_star 추정 ##############
        nth = len(B_p)+len(B_q)
        u_curr = u.iloc[i,nth]  
        u_cand = float(np.random.normal(u_curr,np.sqrt(EmpVar(nth)),size=1))

        r = np.exp(fst(u_cand,B_q,a_l,phi) - fst(u_curr,B_q,a_l,phi))
        values = np.array([u_cand,u_curr])
        p = np.array([r, 1-r])
        if r >= 1:
            B_s = u_cand
        else:
            B_s = float(np.random.choice(values, size=1, p=p))

        u.iloc[i+1,nth] = B_s
        ############## a_l 추정: independent MH ##############
        for lth in range(L):
            curr = a_l[Time*lth]  
            cand = float(np.random.normal(0,sd_a_l,size=1))

            r = np.exp(fa_l(cand,B_p,B_q,B_s) - fa_l(curr,B_p,B_q,B_s))

            tmp = float(np.random.uniform(0,1,size=1))
            if tmp <= r:
                a_l[Time*lth:(Time*(lth+1))] = cand
            else:
                a_l[Time*lth:(Time*(lth+1))] = curr
        ######################################################
        sys.stdout.write('\rRepetition is %d and Iteration is %04s' %(j,i))
        sys.stdout.flush()
    #########################################################################################
    u1 = u.drop(labels=range(0,burn+1),axis=0).reset_index(drop = True)  
       
    B_Mean = u1.mean().values.flatten() 
 
    PostMean.loc[j] = (B_Mean).tolist()
    
    B_SD   = u1.std().values.flatten()  
    
    B_L = B_Mean - z * B_SD
    B_U = B_Mean + z * B_SD
    
 
    Bias.iloc[j] = (B_Mean - params).tolist()
    
 
    MSE.iloc[j] = (B_SD**2 + (Bias.iloc[j])**2).tolist()

 
    T_or_F.iloc[j] = ((params >= B_L) & (params <= B_U))
    
 
    AW.iloc[j] = (B_U - B_L).tolist()

T_or_F = T_or_F*1 

t2 = time.time()
sec = t2-t1

result_list = str(datetime.timedelta(seconds=sec)).split(".")
print("\nTime elapsed:",result_list[0])


# ---

# # Results

# In[1]:


Final_rslt = pd.DataFrame({'True Value': params,
                           'PostMean': PostMean.mean(),
                           'Bias': Bias.mean(),
                           'MSE': MSE.mean(),
                           'CP': (T_or_F.sum()/T_or_F.shape[0]),
                           'AW': AW.mean()})


# ---

# In[13]:


Final_rslt.drop(['bp0','bq0'])


# ---
