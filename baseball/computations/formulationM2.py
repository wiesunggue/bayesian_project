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
aa = pd.read_csv('M2-realdat-u2.csv')
db1=aa['bp1'].values
db1.sort()
db3=aa['bp3'].values
db3.sort()
db4=aa['bp4'].values
db4.sort()
dq1=aa['bq1'].values
dq1.sort()
dq8=aa['bq8'].values
dq8.sort()
dq9=aa['bq9'].values
dq9.sort()
bst=aa['bst'].values
bst.sort()
L = len(aa)
print('name',"LB\t\t\t\t\t","UB\t\t\t\t","Mean\t\t\t\t",'std')
print("db1",db1[int(L*0.025)],db1[int(L*0.975)],np.mean(db1),np.std(db1))
print("db3",db3[int(L*0.025)],db3[int(L*0.975)],np.mean(db3),np.std(db3))
print("db4",db4[int(L*0.025)],db4[int(L*0.975)],np.mean(db4),np.std(db4))
print("dq1",dq1[int(L*0.025)],dq1[int(L*0.975)],np.mean(dq1),np.std(dq1))
print("dq8",dq8[int(L*0.025)],dq8[int(L*0.975)],np.mean(dq8),np.std(dq8))
print("dq9",dq9[int(L*0.025)],dq9[int(L*0.975)],np.mean(dq9),np.std(dq9))
print("bst",bst[int(L*0.025)],bst[int(L*0.975)],np.mean(bst),np.std(bst))

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
matplotlib.rcParams['xtick.major.pad'] = '10'
matplotlib.rcParams['ytick.major.pad'] = '6'
# plt.style.use('default')


# In[429]:


df = pd.read_csv('M2-realdat-u2.csv')


# ---

# In[449]:

A1 = pd.read_csv("M2-realdat-A1.csv")
A1 = A1.drop(columns = "i",axis=1)
K1 = pd.read_csv("M2-readlat-K1.csv")
K1 = K1.drop(columns = "i",axis=1)
z = norm.ppf(.975)

stats_u = df.describe().transpose()[['mean','std']]

B_Mean = stats_u.loc[:,['mean']].values.flatten()
B_SD = stats_u.loc[:,['std']].values.flatten()

B_L = B_Mean - z * B_SD
B_U = B_Mean + z * B_SD


# In[509]:


TABLE = pd.DataFrame({"Mean":B_Mean, "SD":B_SD, "Lower": B_L, "Upper": B_U}, index = stats_u.index)
TABLE.to_csv("TABLE-real-M2.csv")
tb = pd.DataFrame(A1.mean().values, columns=['$a_l$'])
print(A1.mean().values)
tb["$k_l$"] = K1.mean().values

sorted_tb = tb.sort_values(by='$a_l$')  ## a_l 기준으로 오름차순 정렬 !
sorted_tb.head()

# In[487]:


plt.rcParams['figure.figsize'] = [8, 11]
plt.rcParams['lines.linewidth'] = 3
# plt.rcParams['font.size'] = '16'
plt.rc('xtick', labelsize=15)  # Set the axes labels font size

##############################################################################
x = sorted_tb.index.values  ## Index

a_points = sorted_tb.loc[:, "$a_l$"].values.flatten()
k_points = sorted_tb.loc[:, "$k_l$"].values.flatten()
bst_point = TABLE.loc["bst", "Mean"]

bst_a = bst_point * a_points  ## beat_st * a_l

y1 = a_points
y2 = bst_point * a_points
y3 = bst_point * a_points + k_points

# In[518]:


fig, ax = plt.subplots(1, 2, figsize=(18, 7))

x = range(60)
ax0 = ax[0]
ax0.plot(x, y1, label=r'$a_l$')
ax0.set_xticks(np.arange(0, 60, 10))
ax0.xaxis.set_major_locator(MultipleLocator(10))
ax0.xaxis.set_major_formatter('{x:.0f}')
ax0.axes.xaxis.set_ticklabels([])
ax0.legend(markerscale=2., fontsize=18, loc='lower right')
ax0.set_xlabel('Individual', fontsize=16)
ax0.set_ylabel('logit $p$', fontsize=20)

ax1 = ax[1]
ax1.plot(x, y2, label=r'$\beta^* \ a_l$')
ax1.plot(x, y3, label=r'$\beta^* \ a_l+\kappa_l$')
ax1.set_xticks(np.arange(0, 60, 10))
ax1.axes.xaxis.set_ticklabels([])
ax1.legend(markerscale=2., fontsize=15, loc='lower right')
ax1.set_xlabel('Individual', fontsize=16)
ax1.set_ylabel('logit $q$', fontsize=20)
# ax1.set_title('fourth subplot')
plt.subplots_adjust(hspace=0.35)

plt.savefig("rd_eff_ind.pdf", format="pdf", bbox_inches="tight")
plt.savefig("rd_eff_ind.jpg",bbox_inches="tight")
plt.show()

# In[ ]:




