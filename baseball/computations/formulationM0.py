import numpy as np
import pandas as pd

aa = pd.read_csv('M0-realdat-u.csv')
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
L = len(aa)
print('name',"LB\t\t\t\t\t","UB\t\t\t\t","Mean\t\t\t\t\t","std")
print("db1",db1[int(L*0.025)],db1[int(L*0.975)],np.mean(db1),np.std(db1))
print("db3",db3[int(L*0.025)],db3[int(L*0.975)],np.mean(db3),np.std(db3))
print("db4",db4[int(L*0.025)],db4[int(L*0.975)],np.mean(db4),np.std(db4))
print("dq1",dq1[int(L*0.025)],dq1[int(L*0.975)],np.mean(dq1),np.std(dq1))
print("dq8",dq8[int(L*0.025)],dq8[int(L*0.975)],np.mean(dq8),np.std(dq8))
print("dq9",dq9[int(L*0.025)],dq9[int(L*0.975)],np.mean(dq9),np.std(dq9))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

# In[14]:


## 그래프에서 마이너스 기호가 표시되도록 하는 설정
matplotlib.rcParams['axes.unicode_minus'] = False

## Add space between the ticklabels and the axes in matplotlib.
matplotlib.rcParams['xtick.major.pad'] = '10'
matplotlib.rcParams['ytick.major.pad'] = '6'

# In[17]:


df = aa

fig, axes = plt.subplots(4, 4, figsize=(13, 8))
fig.suptitle('Traceplots of parameters (Model0)', fontsize=20, position=(0.5, 1), fontweight="bold")
# fig.subplots_adjust(hspace=1,wspace=0.5)

for col, ax in zip(df.columns, axes.flatten()):
    ax.plot(df.index, df[col])

    ax.set_xticks(np.linspace(start=0, stop=df.shape[0], num=6))

    ax.set_ylabel(r"$\beta$" + col[1:3], fontsize=12, rotation="horizontal", labelpad=10)
    #     show_y = np.linspace(start=df[col].min(), stop=df[col].max(), num=4)
    show_y = np.linspace(start=df.min(axis=0)[col], stop=df.max(axis=0)[col], num=4)
    ax.set_yticks(np.round(show_y, 2))
    ax.tick_params(axis='y', labelsize=10)

fig.delaxes(axes[3][2]);
fig.delaxes(axes[3][3])
plt.tight_layout()
plt.show()

