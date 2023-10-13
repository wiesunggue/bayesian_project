#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[5]:


df = pd.read_csv("MLB datasets-2021-demeaned.csv")
L = 60; Time = 6 

m = df[["m"]].values.flatten()
x1 = df[["x_1"]].values.flatten()
x2 = df[["x_2"]].values.flatten()
Z_demd = df.drop(['Team','Player','month','m','x_1','x_2',"Pull%","LD%", "GB%", "Oppo%"], axis=1)
Z_demd.head()


# ---

# In[6]:


z_p = Z_demd.loc[:,["Intercept","WPA", "BABIP","BB/K"]]
z_q = Z_demd.loc[:,["Intercept","WPA", "FB%", "HR/FB"]]
##############################################
## Initial values for each parameter
B_p = [0,0.06,3.37,0.14]
B_q = [0,0.72,0.60,0.90]
B_s = [1.2]
params = B_p + B_q + B_s
##############################################
d = len(params)

phi=100


# In[7]:


sd_a_l = 1

np.random.seed(0)
a_l = np.random.normal(0,sd_a_l,size = L)
a_l = np.repeat(a_l, Time, axis=0) 


sd_k_l = 1

np.random.seed(1)
k_l = np.random.normal(0,sd_k_l,size = L)
k_l = np.repeat(k_l, Time, axis=0) 


# ---

# In[8]:


def fp(tta,bps,a_l,phi):
    bps[k] = tta
    Z_Bp = z_p.dot(np.diag(bps))
    logit_p = np.array(Z_Bp.assign(a_l=a_l).sum(axis=1))
    ests = (tta * (z_p.iloc[:, [k]].to_numpy().flatten()) * x1 - m*ln(1+np.exp(logit_p))).sum()-(tta**2)/(2*(phi**2))
    return ests


def fq(tta,bqs,bst,a_l,k_l,phi):
    bqs[k] = tta
    Z_Bq = z_q.dot(np.diag(bqs))
    logit_q = np.array(Z_Bq.assign(addcol=bst*a_l+k_l).sum(axis=1))
    ests = (tta * (z_q.iloc[:, [k]].to_numpy().flatten()) * x2 - x1*ln(1+np.exp(logit_q))).sum()-(tta**2)/(2*(phi**2))
    return(ests)


def fst(tta,bqs,a_l,k_l,phi):
    Z_Bq = z_q.dot(np.diag(bqs))
    logit_q = np.array(Z_Bq.assign(addcol=tta*a_l+k_l).sum(axis=1))
    ests = (tta*a_l*x2 - x1*ln(1+np.exp(logit_q))).sum()-(tta**2)/(2*(phi**2))
    return(ests)


def fa_l(tta,bps,bqs,bst,k_l):  
    now_l = list(range(Time*lth, Time*(lth+1)))
    
    Z_Bp_l = z_p.iloc[now_l,:].dot(np.diag(bps))
    logit_pp = np.array(Z_Bp_l.assign(addcol=tta).sum(axis=1))
   
    Z_Bq_l = z_q.iloc[now_l,:].dot(np.diag(bqs))
    logit_qq = np.array(Z_Bq_l.assign(addcol=bst*tta + k_l[lth]).sum(axis=1))
    
    ests = (tta*(x1[now_l]+bst*x2[now_l]) - m[now_l]*ln(1+np.exp(logit_pp))
            - x1[now_l]*ln(1+np.exp(logit_qq))).sum()    
    return(ests)


def fk_l(tta,bps,bqs,bst,a_l):  
    now_l = list(range(Time*lth, Time*(lth+1)))
    
    Z_Bp_l = z_p.iloc[now_l,:].dot(np.diag(bps))
    logit_pp = np.array(Z_Bp_l.assign(addcol=a_l[now_l]).sum(axis=1))
   
    Z_Bq_l = z_q.iloc[now_l,:].dot(np.diag(bqs))
    logit_qq = np.array(Z_Bq_l.assign(addcol=bst*a_l[now_l] + tta).sum(axis=1))
    
    ests = (tta*(x2[now_l]) - m[now_l]*ln(1+np.exp(logit_pp))
            - x1[now_l]*ln(1+np.exp(logit_qq))).sum() 
    return(ests)


# In[9]:


s = 2.4/math.sqrt(d-2) 

np.random.seed(10)
eps = np.random.uniform(0,1,size=d)*0.05
Var_0  = np.random.uniform(5,20,size=d)
def EmpVar(nth):
    if i == 0:
        rslt = Var_0[nth]
    elif i == 1:
        rslt = s*eps[nth]
    else: 
        rslt = s*np.var(u.iloc[0:(i-1),nth]) + s*eps[nth]
    return(rslt)


# ---

# In[10]:


itr = 6
burn = 1
thin = 1


# ---

# In[11]:


u_colnames = ["bp" + str(num1) for num1, in zip([0,1,3,4])]+["bq" + str(num1) for num1, in zip([0,1,8,9])]+["bst"]
A_colnames = ["a" + str(num1) for num1, in zip(list(range(1, L+1)))]
K_colnames = ["k" + str(num1) for num1, in zip(list(range(1, L+1)))]


# In[12]:


u = np.zeros((itr+1, d))
u = pd.DataFrame(u, columns = u_colnames)
u.iloc[0,:] = params
u.head(3)
accpt_n_l = np.zeros(L)
accpt_n_k = np.zeros(L)
# In[13]:


A = np.zeros((itr+1, L))
A = pd.DataFrame(A, columns = A_colnames)
A.iloc[0,:] = a_l[list(range(0, 360, Time))] 

K = np.zeros((itr+1, L))
K = pd.DataFrame(K, columns = K_colnames)
K.iloc[0,:] = k_l[list(range(0, 360, Time))] 


# ---

# In[17]:


t1 = time.time()
######################################################################################
for i in range(itr):
    np.random.seed(i)
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
        u_curr = u.iloc[i,nth] ## current value
        u_cand = float(np.random.normal(u_curr,np.sqrt(EmpVar(nth)),size=1))

        B_q2 = B_q      
        B_q2[k] = u_cand
        
        r = np.exp(fq(u_cand,B_q2,B_s,a_l,k_l,phi) - fq(u_curr,B_q,B_s,a_l,k_l,phi))
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
    
    r = np.exp(fst(u_cand,B_q,a_l,k_l,phi) - fst(u_curr,B_q,a_l,k_l,phi))
    values = np.array([u_cand,u_curr])
    p = np.array([r, 1-r])
    if r >= 1:
        B_s = u_cand
    else:
        B_s = float(np.random.choice(values, size=1, p=p))
        
    u.iloc[i+1,nth] = B_s
     
    for lth in range(L):
        curr = A.iloc[i,lth]  
        cand = float(np.random.normal(0,sd_a_l,size=1))
        
        r = np.exp(fa_l(cand,B_p,B_q,B_s,k_l) - fa_l(curr,B_p,B_q,B_s,k_l))
       
        tmp = float(np.random.uniform(0,1,size=1))
        if tmp <= r:
            A.iloc[i+1,lth] = cand
            if itr >= burn:    
                accpt_n_l[lth] = accpt_n_l[lth]+1
        else:
            A.iloc[i+1,lth] = curr         
    a_l = np.repeat(A.iloc[i+1,:], Time).array
    
 
    for lth in range(L):
        curr = K.iloc[i,lth]  
        cand = float(np.random.normal(0,sd_k_l,size=1))
        
        r = np.exp(fk_l(cand,B_p,B_q,B_s,a_l) - fk_l(curr,B_p,B_q,B_s,a_l))
       
        tmp = float(np.random.uniform(0,1,size=1))
        if tmp <= r:
            K.iloc[i+1,lth] = cand
            if itr >= burn:   
                accpt_n_k[lth] = accpt_n_k[lth]+1
        else:
            K.iloc[i+1,lth] = curr
    k_l = np.repeat(K.iloc[i+1,:], Time).array
    
    sys.stdout.write('\rIteration is %d' %i)  
    sys.stdout.flush()
######################################################################################
u1 = u.drop(labels=range(0,burn+1),axis=0).reset_index(drop = True) ## burn-in     
u2 = u1.drop(['bp0', 'bq0'], axis=1)

A1 = A.drop(labels=range(0,burn+1),axis=0).reset_index(drop = True) ## burn-in     
K1 = K.drop(labels=range(0,burn+1),axis=0).reset_index(drop = True) ## burn-in     


t2 = time.time()
sec = t2-t1

result_list = str(datetime.timedelta(seconds=sec)).split(".")
print("\nTime elapsed:",result_list[0])


# ---

# # Results

# In[21]:


z = norm.ppf(.975) 

stats_u = u1.describe().transpose()[['mean','std']]

B_Mean = stats_u.loc[:,['mean']].values.flatten()
B_SD = stats_u.loc[:,['std']].values.flatten()

B_L = B_Mean - z * B_SD
B_U = B_Mean + z * B_SD


# In[509]:


TABLE = pd.DataFrame({"Mean":B_Mean, "SD":B_SD, "Lower": B_L, "Upper": B_U}, index = stats_u.index)
TABLE.to_csv("TABLE-real-M2.csv")
u2.to_csv("M2-realdat-u2.csv")
A1.to_csv("M2-realdat-A1'.csv")
K1.to_csv("M2-readlat-K1.csv")
print(A1.mean().values)
TABLE


# ---

# # Traceplots of all parameters

# In[426]:


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, IndexLocator, FuncFormatter
from matplotlib.dates import MonthLocator, DateFormatter


# In[427]:


## 그래프에서 마이너스 기호가 표시되도록 하는 설정
matplotlib.rcParams['axes.unicode_minus'] = False

## Add space between the ticklabels and the axes in matplotlib.
matplotlib.rcParams['xtick.major.pad']='10'
matplotlib.rcParams['ytick.major.pad']='6'
# plt.style.use('default')


# In[429]:


df = u2

fig, axes = plt.subplots(4, 4, figsize=(13,8))
fig.suptitle('Traceplots of parameters', fontsize=20, position=(0.5, 1),fontweight="bold")
# fig.subplots_adjust(hspace=1,wspace=0.5)

for col, ax in zip(df.columns, axes.flatten()):
    ax.plot(df.index, df[col])
    
    ax.set_xticks(np.linspace(start=0, stop=df.shape[0], num=6))
    
    ax.set_ylabel(r"$\beta$"+col[1:3], fontsize=12, rotation="horizontal",labelpad=10)
 
    show_y = np.linspace(start=df.min(axis=0)[col], stop=df.max(axis=0)[col], num=4)
    ax.set_yticks(np.round(show_y,2))
    ax.tick_params(axis='y', labelsize=10)
    
fig.delaxes(axes[3][1]);fig.delaxes(axes[3][2]);fig.delaxes(axes[3][3])  
plt.tight_layout()
plt.show()


# ---

# In[449]:


tb = pd.DataFrame(A1.mean().values, columns = ['$a_l$'])
tb["$k_l$"] = K1.mean().values

sorted_tb = tb.sort_values(by='$a_l$') ## a_l 기준으로 오름차순 정렬 !
sorted_tb.head()


# In[487]:


plt.rcParams['figure.figsize'] = [8,11]
plt.rcParams['lines.linewidth'] = 3
# plt.rcParams['font.size'] = '16'
plt.rc('xtick', labelsize = 15) # Set the axes labels font size

##############################################################################
x = sorted_tb.index.values ## Index 

a_points = sorted_tb.loc[:,"$a_l$"].values.flatten()  
k_points = sorted_tb.loc[:,"$k_l$"].values.flatten() 
bst_point = TABLE.loc["bst","Mean"] 

bst_a = bst_point * a_points ## beat_st * a_l  

y1 = a_points
y2 = bst_point * a_points
y3 = bst_point * a_points + k_points


# In[518]:


fig, ax = plt.subplots(1,2,figsize=(18,7))

x=range(60)
ax0= ax[0]
ax0.plot(x,y1,label = r'$a_l$')
ax0.set_xticks(np.arange(0, 60, 10))
ax0.xaxis.set_major_locator(MultipleLocator(10)) 
ax0.xaxis.set_major_formatter('{x:.0f}') 
ax0.axes.xaxis.set_ticklabels([])
ax0.legend(markerscale=2., fontsize=18, loc='lower right')
ax0.set_xlabel('Individual', fontsize=16)
ax0.set_ylabel('logit $p$', fontsize=20)

ax1= ax[1]
ax1.plot(x, y2, label = r'$\beta^* \ a_l$')
ax1.plot(x, y3, label = r'$\beta^* \ a_l+\kappa_l$')
ax1.set_xticks(np.arange(0, 60, 10))
ax1.axes.xaxis.set_ticklabels([])
ax1.legend(markerscale=2., fontsize=15,loc='lower right')
ax1.set_xlabel('Individual', fontsize=16)
ax1.set_ylabel('logit $q$', fontsize=20)
# ax1.set_title('fourth subplot')
plt.subplots_adjust(hspace=0.35)

plt.savefig("rd_eff_ind.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[ ]:




