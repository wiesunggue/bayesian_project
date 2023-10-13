from generater import ZIP,Likelihood,correction_fractional,correction_AIBF,intrinsic_prior
from scipy.integrate import quad,dblquad
import numpy as np
from decimal import Decimal
import time
import pandas as pd
w_default,l,N=0.5,4,30
w=0.5
infi=500
s = 10
iprior = lambda l: intrinsic_prior(w,l,s,w_default)

c=quad(iprior,0,infi,limit=100)[0]
print("normalize constant : ",c)


rep = 30
start=time.time()
n_start = start
w_start = start
st = '========== Start Simulation =========='
print(st)
print('-'*len(st))
RD_df = pd.DataFrame()
RD_df_div = pd.DataFrame()
RD_df_mul3 = pd.DataFrame()
bayefactor = pd.DataFrame()
for w in [0.1,0.3,0.5,0.7,0.9]:
    w_start = time.time()
    print(f'{w} iteration start')
    print('-'*len(st))
    A = []
    B = []
    C = []
    bf = []
    for i in range(rep):
        data = ZIP(w, l, N)  # data 50개 생성
        Lh = lambda w, l: Likelihood(w=w, l=l, data=data, prior=iprior)
        L1 = lambda l: Likelihood(w_default, l, data)  # 함수에서 data, w_default로 고정 => l만 입력받음
        L2 = lambda w, l: Likelihood(w, l, data)  # 함수에서 data만 고정 => l,w만 입력받음
        if (i+1)%10==0:
            n_end = time.time()
            print(f'{i+1}/{rep}번 반복 소요시간 : {n_end-n_start:.4f}s')
            n_start=n_end
        m1 = quad(L1, 0, infi)[0] # 적분하는 함수 l = 0~inf
        #print("m1",m1)
        m2 = dblquad(L2, 0, infi, 0, 1)[0] # 이중적분 하는 함수 w=0~1 l = 0~inf
        #print("m2",m2)
        mi2 = dblquad(Lh,0,infi,0,1)[0]
        #print("mi2",mi2)
        BI_S = mi2/m1
        BI_SperC = BI_S/c
        BI = m2/m1*correction_AIBF(w_default,data)+10**-7
        RD = abs(BI_S-BI)/BI
        RD_perC = abs(BI_SperC-BI)/BI
        A.append(RD)
        B.append(RD_perC)
        C.append(abs(3*BI_S-BI)/BI)
        bf.append((BI_S,BI_SperC,BI))
    bf = pd.DataFrame(bf, columns=[f"w={w} Bayes Factor", f"w={w} Bayes Factor with Norm", f"w={w} bf*correction"])
    bayefactor=pd.concat([bayefactor,bf],axis=1)
    RD_df[f'w = {w}']=A
    RD_df_div[f'w = {w}']=B
    RD_df_mul3[f'w = {w}']=C
    end = time.time()
    print(f'w={w} 반복 소요시간 : {end-w_start:.4f}s')
    
print(f'총 소요 시간:{time.time()-start:.4f}s ')
bayefactor.to_csv('bayesfactor result.csv')

RD_df.to_csv('RD.csv')
RD_df_div.to_csv("RD with norm.csv")
RD_df_mul3.to_csv("RD mul3.csv") # 이상하게 비슷한 놈