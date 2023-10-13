from generater import ZIP,Likelihood,correction_fractional,correction_AIBF
import matplotlib.pyplot as plt
from scipy.integrate import quad,dblquad
import numpy as np
import pandas as pd
import time

w_default = 0.5 # 검증을 위해서 사용할 변수
start = time.time()
n_start = start
pn = 50
for w in [0.1,0.3,0.5,0.7,0.9]: # 데이터 생성에 사용할 변수를 반복적으로 대입하는 과정
    l,N=4,30 # 람다와 데이터의 개수
    rep = 1000 #반복횟수
    arr = [0] * rep # 미리 저장할 배열 선언
    print('-'*50)
    print(f'Start iteration w = {w}')
    for i in range(rep):
        if (i+1)%pn==0:
            n_end = time.time()
            print("{}단계 {}/{}번 반복, {}번 반복 소요시간 : {:.2f}".format(w,i+1,rep,pn,n_end-n_start))
            n_start = n_end
        data = ZIP(w,l,N)  # 데이터 생성
        L1 = lambda l: Likelihood(w_default, l, data) #함수에서 data, w_default로 고정 => l만 입력받음
        L2 = lambda w, l: Likelihood(w, l, data) # 함수에서 data만 고정 => l,w만 입력받음
        m1 = quad(L1, 0, np.inf) # 적분하는 함수 l = 0~inf
        m2 = dblquad(L2, 0, np.inf, 0, 1) # 이중적분 하는 함수 w=0~1 l = 0~inf
        arr[i] = (m1[0],m2[0],correction_AIBF(w_default,data),correction_fractional(w_default,data))

    df = pd.DataFrame(arr)
    df.to_csv(f'{N}save_data({w}).csv') # csv파일로 저장(총 5개 생성)
end = time.time()

print("총 연산 시간 : {:.2f}".format(end-start))
