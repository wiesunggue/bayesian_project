from decimal import Decimal
import decimal
import time
from functools import partial,reduce
start=time.time()
import sympy as sym
from sympy import Integral, symbols
end=time.time()
print("sympy : 모듈 불러오는 시간: {:.4f}(s)".format(end-start))
start=time.time()
import scipy as sc
from scipy.optimize import bisect,fsolve
from scipy.integrate import quad,nquad,dblquad,tplquad
end=time.time()
print("scipy : 모듈 불러오는 시간: {:.4f}(s)".format(end-start))

import numpy as np
from numpy.random import uniform,gamma,normal,exponential
from math import exp,log10,sqrt
import matplotlib.pyplot as plt

sigma = Decimal(10)
def set_parameter(aa,bb,cc):
    # a1 = alpha_1
    # b1 = beta_1
    # 주의 b1은 var까지 더해줘야 해서 정의를 다르게 함(넣을 때 1/(b1+sum(var))을 사용)
    a1,b1=Decimal(aa**2/sigma**2),Decimal(aa/sigma**2)
    a2,b2=Decimal(bb**2/sigma**2),Decimal(bb/sigma**2)
    a3,b3=Decimal(cc**2/sigma**2),Decimal(cc/sigma**2)
    print("a1 : {:.4f} b1 : {:.4f} a2 : {:.10f} b2 : {:.4f} a3 : {:.4f} b3 : {:.4f}".format(a1,b1,a2,b2,a3,b3))
    return a1,b2,a2,b2,a3,b3
def set_parameter_mcmc(aa,bb,cc):
    # a1 = alpha_1
    # b1 = beta_1
    # 주의 b1은 var까지 더해줘야 해서 정의를 다르게 함(넣을 때 1/(b1+sum(var))을 사용)
    a1,b1=Decimal(aa**2/sigma**2),Decimal(sigma**2/aa)
    a2,b2=Decimal(bb**2/sigma**2),Decimal(bb/sigma**2)
    a3,b3=Decimal(cc**2/sigma**2),Decimal(cc/sigma**2)
    print("a1 : {:.4f} b1 : {:.4f} a2 : {:.10f} b2 : {:.4f} a3 : {:.4f} b3 : {:.4f}".format(a1,b1,a2,b2,a3,b3))
    return a1,b2,a2,b2,a3,b3
mul = lambda x,y:x*y # 곱하기 함수
add = lambda x,y:x+y # 더하기 함수

M = lambda aa,bb,cc,d1,d2:(Decimal(quad(lambda y : (-bb*Decimal(y)).exp()/(aa+cc*Decimal(y)),d1,d2)[0]),0)

# gamma distribution의 정의
Gamma_dist = lambda x,alpha,beta : x**alpha*exp(-beta*x)*beta**alpha/gamma(alpha)

#Prior_A = partial(Gamma_dist,alpha=a1,beta=b1)
#Prior_B = partial(Gamma_dist,alpha=a2,beta=b2)
#Prior_C = partial(Gamma_dist,alpha=a3,beta=b3)
#Prior_d1 = lambda d1 : 1/(1+d1)**2 if d1>0 else 0 # d1>0
#Prior_d2 = lambda d1,d2 : 1/(d2-d1) if d2>d1 else 0 # d2>d1

# Likelihood
### variables는 반드시 numpy형 배열을 넣어야 한다
# variables = np.array([1,2,3,4])
Likelihood = lambda aa,bb,cc,d1,d2,variables : 1/M(aa,bb,cc,d1,d2)[0]**len(variables)\
                                            *exp(-aa*sum(variables))\
                                            *(exp(-(len(variables)*bb+cc*sum(variables))*d1)-exp(-(len(variables)*bb+cc*sum(variables))*d2))\
                                            /reduce(mul,[i*cc+bb for i in variables]) # b+cx를 변수의 개수번 곱하는 함수

# Posterior Distribution
# 그냥 곱으로 표현하지 않고 처음부터 다시 적는게 나을 수도...
Posterior1=lambda aa,bb,cc,d1,d2,variables : Likelihood(aa,bb,cc,d1,d2,variables)*Prior_A(aa)*Prior_B(bb)*Prior_C(cc)*Prior_d1(d1)*Prior_d2(d1,d2)
#print(rhtexp(10,aa,bb,cc,d1,d2))


# full conditional distributions
# 공통 연산인 부분 common1은 b,c부분, common2는 d1,d2부분
common1 = lambda aa,bb,cc,d1,d2,variables : 10**reduce(add,[(((-(bb+cc*i)*d1).exp()-(-(bb+cc*i)*d2).exp())/(bb+cc*i)).log10() for i in variables])
common2 = lambda aa,bb,cc,d1,d2,variables : 10**reduce(add,[((-(bb+cc*i)*d1).exp()-(-(bb+cc*i)*d2).exp()).log10() for i in variables])

P_A = lambda aa,bb,cc,d1,d2,variables : 1/M(aa,bb,cc,d1,d2)[0]**Decimal(len(variables))*aa**(1-Decimal(1))*(-aa*(reduce(add,variables)+Decimal(0.8))).exp() # ok
P_B = lambda aa,bb,cc,d1,d2,variables : 1/M(aa,bb,cc,d1,d2)[0]**Decimal(len(variables))*common1(aa,bb,cc,d1,d2,variables)*bb**(1/sigma-Decimal(1))*(-1/sigma*bb).exp()
P_C = lambda aa,bb,cc,d1,d2,variables : 1/M(aa,bb,cc,d1,d2)[0]**Decimal(len(variables))*common1(aa,bb,cc,d1,d2,variables)*cc**(1/sigma-1)*(-1/sigma*cc).exp()
P_d1 = lambda aa,bb,cc,d1,d2,variables : 1/M(aa,bb,cc,d1,d2)[0]**Decimal(len(variables))*common1(aa,bb,cc,d1,d2,variables)*(1/(1+d1))**2
P_d2 = lambda aa,bb,cc,d1,d2,variables : 1/M(aa,bb,cc,d1,d2)[0]**Decimal(len(variables))*common1(aa,bb,cc,d1,d2,variables)/(d2-d1)

def update(ratio,tmp,before):
    if ratio>=1: # r이 1보다 크면 업데이트
        return tmp
    else:
        return tmp if uniform()<ratio else before # ratio의 확률로 업데이트


def MCMC_independent(aa, bb, cc, d1, d2, var,radius,cnt_list):
    """
    현재의 문제점
    1. common의 나눗셈 연산이 아닌 식을 좀 더 축약해서 써야 함(연산 순서 변경)
    2. b,c 가 0으로 => alpha,beta가 0에 거의 가까워짐(10^-100수준)=> gamma의 값이 0 추출됨
    ratio_b, ratio_c도 0으로 가려고 함
    ---------------------
    MCMC 1회 연산하는 함수
    :param aa,bb,cc,d1,d2: 초기값
    :param var: 뽑은 데이터 셋

    :return: 연산이 끝난 후의 값
    """
    N = len(var)  # 생성한 데이터의 개수
    a_cnt,b_cnt,c_cnt,d1_cnt,d2_cnt = cnt_list
    # 업데이트 A
    a1, b1, a2, b2, a3, b3 = set_parameter_mcmc(aa, bb, cc)  # alpha beta 업데이트
    a_tmp = Decimal(np.random.gamma(a1, 1 / (b1 + sum(var))))  # 감마에서 임시값 뽑기
    print('a_tmp', a_tmp)
    ratio_a = Decimal(M(aa, bb, cc, d1, d2)[0] / M(a_tmp, bb, cc, d1, d2)[0]) ** N  # r구하기
    a_next = update(ratio_a, a_tmp, aa)

    # 업데이트 B
    a1, b1, a2, b2, a3, b3 = set_parameter_mcmc(a_next, bb, cc)
    b_tmp = Decimal(np.random.gamma(a2, b2))
    # print("b_tmp 의 감마 추출값 : ",a2,b2,b_tmp)
    ratio_b = Decimal(M(a_next, bb, cc, d1, d2)[0] / M(a_next, b_tmp, cc, d1, d2)[0]) ** N \
              * reduce(mul, [((-(b_tmp + cc * i) * d1).exp() - (-(b_tmp + cc * i) * d2).exp()) / (
                (-(bb + cc * i) * d1).exp() - (-(bb + cc * i) * d2).exp()) * (bb + cc * i) / (b_tmp + cc * i) for i in
                             var])
    b_next = update(ratio_b, b_tmp, bb)

    # 업데이트 C
    a1, b1, a2, b2, a3, b3 = set_parameter_mcmc(a_next, b_next, cc)
    c_tmp = Decimal(np.random.gamma(a3, b3))
    ratio_c = Decimal(M(a_next, b_next, cc, d1, d2)[0] / M(a_next, b_next, c_tmp, d1, d2)[0]) ** N \
              * reduce(mul, [((-(b_next + c_tmp * i) * d1).exp() - (-(b_next + c_tmp * i) * d2).exp()) / (
                (-(b_next + cc * i) * d1).exp() - (-(b_next + cc * i) * d2).exp()) * (b_next + cc * i) / (b_next + c_tmp * i) for i in
                             var])
    # *(common1(a_next,b_next,c_tmp,d1,d2,var)/common1(a_next,b_next,cc,d1,d2,var))
    c_next = update(ratio_c, c_tmp, cc)

    # 업데이트 d1

    a1, b1, a2, b2, a3, b3 = set_parameter_mcmc(a_next, b_next, c_next)
    d1_tmp = Decimal((1 + d2) / (1 + d2 * Decimal(1 - uniform())) - Decimal(1))  # 역함수 활용
    print("---update state d1---\n a_next : {:.4f} b_next : {:.4f} c_next : {:.4f} d1_tmp : {:.4f} d2 : {:.4f}".format(
        a_next, b_next, c_next, d1_tmp, d2))
    # print("문제가 제일 많이 생기는 common2 : {}".format(common2(a_next,b_next,c_next,d1_tmp,d2,var)))
    ratio_d1 = Decimal(M(a_next, b_next, c_next, d1, d2)[0] / M(a_next, b_next, c_next, d1_tmp, d2)[0]) ** N \
               * reduce(mul, [((-(b_next + c_next * i) * d1_tmp).exp() - (-(b_next + c_next * i) * d2).exp()) / (
                (-(b_next + c_next * i) * d1).exp() - (-(b_next + c_next * i) * d2).exp()) for i in var])*(1+d1)**2/(1+d1_tmp)**2
    # *common2(a_next,b_next,c_next,d1_tmp,d2,var)/common2(a_next,b_next,c_next,d1,d2,var)
    d1_next = update(ratio_d1, d1_tmp, d1)

    # 업데이트 d2
    d2_tmp = Decimal(uniform(d2 - radius, d2 + radius))  # 수정예정
    ratio_d2 = Decimal(M(a_next, b_next, c_next, d1_next, d2)[0] / M(a_next, b_next, c_next, d1_next, d2_tmp)[0]) ** N \
               * reduce(mul, [((-(b_next + c_next * i) * d1_next).exp() - (-(b_next + c_next * i) * d2_tmp).exp()) / (
                (-(b_next + c_next * i) * d1_next).exp() - (-(b_next + c_next * i) * d2).exp()) for i in var])
    # *common2(a_next,b_next,c_next,d1_next,d2_tmp,var)/common2(a_next,b_next,c_next,d1_next,d2,var)
    d2_next = update(ratio_d2, d2_tmp, d2)
    d2_next = d2_next if d2_next > d1_next and d2_next < 2 else d2
    #print('ratio a : {:.4f} ratio b : {:.4f} ratio c : {:.4f} ratio d1 : {:.4f} ratio d2 : {:.4f}'.format(ratio_a,ratio_b,ratio_c,ratio_d1,ratio_d2))
    if a_next!=aa:
        a_cnt+=1
    if b_next!=bb:
        b_cnt+=1
    if c_next!=cc:
        c_cnt+=1
    if d1_next!=d1:
        d1_cnt+=1
    if d2_next!=d2:
        d2_cnt+=1
    cnt_list = [a_cnt,b_cnt,c_cnt,d1_cnt,d2_cnt]
    return a_next, b_next, c_next, d1_next, d2_next,cnt_list

# 초기값을 설정해주면 알아서 연산
def MCMC_solve(initial_value,var,N_size=10000,drop=1000,sim_print=False):
    '''
    MCMC 구현하는 함수
    :param initial_value:
    aa,bb,cc,d1,d2를 튜플로 입력
    :param var:
    랜덤생성한 데이터 셋
    :param N_size:
    반복횟수
    :param drop:
    초기 시행은 버리는데 버릴 시점 ex)drop=1000, N_size=10000이면 1001번부터 10000번까지의 계산 결과를 활용하여 parameter의 평균을 구함
    :param sim_print:
    시뮬레이션을 프린트할지 여부를 출력
    :return:
    (aa,bb,cc,d1,d2)의 평균값
    '''
    aa,bb,cc,d1,d2,radius = [Decimal(i) for i in initial_value] # decimal 로 형변환
    var = [Decimal(i) for i in var]
    param_dict ={"a_list":[aa],"b_list":[bb],"c_list":[cc],"d1_list":[d1],"d2_list":[d2]}
    cnt_list = [0,0,0,0,0]
    for epoch in range(N_size):
        if epoch%100==0:
            print('{}회 실행'.format(epoch))
        if sim_print==True:
            print("iter : {}\n aa의 현재 : {} 평균 : {}\n bb 현재 : {} 평균 : {}\n cc 현재 : {} 평균 : {}\n d1 현재 : {} 평균 : {}\n d2 현재 : {} 평균 : {}"\
                  .format(epoch,aa,np.mean(param_dict['a_list'])\
                          ,bb,np.mean(param_dict['b_list'])\
                          ,cc,np.mean(param_dict['c_list'])\
                          ,d1,np.mean(param_dict['d1_list'])\
                          ,d2,np.mean(param_dict['d2_list'])))
        # 파라메터 업데이트
        aa,bb,cc,d1,d2,cnt_list = MCMC_independent(aa,bb,cc,d1,d2,var,radius,cnt_list)

        # 파라메터 저장
        param_dict['a_list'].append(aa)
        param_dict['b_list'].append(bb)
        param_dict['c_list'].append(cc)
        param_dict['d1_list'].append(d1)
        param_dict['d2_list'].append(d2)

    # 결과 drop번째부터 계산
    ans = (np.mean(param_dict['a_list'][drop:]),np.mean(param_dict['b_list'][drop:]),np.mean(param_dict['c_list'][drop:]),np.mean(param_dict['d1_list'][drop:]),np.mean(param_dict['d2_list'][drop:]))
    return ans,param_dict,cnt_list

def metropolis_Hastings(aa,bb,cc,d1,d2,var,radius,normal_set,hyper_param,cnt_list):
    a_cnt,b_cnt,c_cnt,d1_cnt,d2_cnt = cnt_list
    """
    MCMC 1회 연산하는 함수
    :param aa,bb,cc,d1,d2: 초기값
    :param var: 뽑은 데이터 셋

    :return: 연산이 끝난 후의 값
    """
    a_mu,a_var,b_mu,b_var,c_mu,c_var=normal_set

    a1,b1,a2,b2,a3,b3 = hyper_param
    N = len(var)  # 생성한 데이터의 개수

    # 업데이트 A
    a_tmp = Decimal(np.random.normal(a_mu,a_var))  # 정규 분포에서 임시값 뽑기
    while a_tmp<0:
        a_tmp = Decimal(np.random.normal(a_mu, a_var))
    #print('a_tmp',a_tmp,aa,a_mu,a_var)
    ratio_a = Decimal(M(aa, bb, cc, d1, d2)[0] / M(a_tmp, bb, cc, d1, d2)[0]) ** N*(a_tmp/aa)**(a1-Decimal(1))*((sum(var)+b1)*(aa-a_tmp)).exp()  # r구하기
    #print('a prob',ratio_a)

    a_next = update(min(1,ratio_a), a_tmp, aa)

    # 업데이트 B
    b_tmp = Decimal(np.random.normal(b_mu, b_var))
    while b_tmp<0:
        b_tmp = Decimal(np.random.normal(b_mu, b_var))
    #print('b_tmp',b_tmp,b_mu,b_var)
    ratio_b = Decimal(M(a_next, bb, cc, d1, d2)[0] / M(a_next, b_tmp, cc, d1, d2)[0]) ** N \
            * reduce(mul, [((-(b_tmp + cc * i) * d1).exp() - (-(b_tmp + cc * i) * d2).exp()) / (
            (-(bb + cc * i) * d1).exp() - (-(bb + cc * i) * d2).exp()) * (bb + cc * i) / (b_tmp + cc * i) for i in var]) * (b_tmp/bb)**(a2-Decimal(1))*(b2*(bb-b_tmp)).exp()
    #print('b prob', ratio_b)
    b_next = update(min(1,ratio_b), b_tmp, bb)

    # 업데이트 C
    c_tmp = Decimal(np.random.normal(c_mu, c_var))
    while c_tmp<0:
        c_tmp = Decimal(np.random.normal(c_mu, c_var))

    #print('c_tmp',c_tmp,c_mu,c_var)
    ratio_c = Decimal(M(a_next, b_next, cc, d1, d2)[0] / M(a_next, b_next, c_tmp, d1, d2)[0]) ** N \
            * reduce(mul, [((-(b_next + c_tmp * i) * d1).exp() - (-(b_next + c_tmp * i) * d2).exp()) / (
            (-(b_next + cc * i) * d1).exp() - (-(b_next + cc * i) * d2).exp()) * (b_next + cc * i) / (b_next + c_tmp * i) for i in var]) *(c_tmp/cc)**(a3-Decimal(1)) *(b3*(cc-c_tmp)).exp()
    #print('c prob', ratio_c)
    c_next = update(min(1,ratio_c), c_tmp, cc)

    d1_tmp = Decimal((1 + d2) / (1 + d2 * Decimal(1 - uniform())) - Decimal(1))
    while d1_tmp<0:
        d1_tmp = Decimal((1 + d2) / (1 + d2 * Decimal(1 - uniform())) - Decimal(1))  # 역함수 활용
    #print('d1_tmp',d1_tmp)
    ratio_d1 = Decimal(M(a_next, b_next, c_next, d1, d2)[0] / M(a_next, b_next, c_next, d1_tmp, d2)[0]) ** N \
            * reduce(mul, [((-(b_next + c_next * i) * d1_tmp).exp() - (-(b_next + c_next * i) * d2).exp()) / (
            (-(b_next + c_next * i) * d1).exp() - (-(b_next + c_next * i) * d2).exp()) for i in var])*(1+d1)**2/(1+d1_tmp)**2
    d1_next = update(min(1,ratio_d1), d1_tmp, d1)

    # 업데이트 d2
    d2_tmp = Decimal(uniform(d2 - radius, d2 + radius))  # 수정예정
    while d2_tmp>2 or d2_tmp<d1_next:
        d2_tmp = Decimal(uniform(d2 - radius, d2 + radius))  # 수정예정

    ratio_d2 = Decimal(M(a_next, b_next, c_next, d1_next, d2)[0] / M(a_next, b_next, c_next, d1_next, d2_tmp)[0]) ** N \
            * reduce(mul, [((-(b_next + c_next * i) * d1_next).exp() - (-(b_next + c_next * i) * d2_tmp).exp()) / (
            (-(b_next + c_next * i) * d1_next).exp() - (-(b_next + c_next * i) * d2).exp()) for i in var])
    d2_next = update(min(1,ratio_d2), d2_tmp, d2)



    #print('ratio a : {:.4f} ratio b : {:.4f} ratio c : {:.4f} ratio d1 : {:.4f} ratio d2 : {:.4f}'.format(ratio_a,ratio_b,ratio_c,ratio_d1,ratio_d2))
    if a_next!=aa:
        a_cnt+=1
    if b_next!=bb:
        b_cnt+=1
    if c_next!=cc:
        c_cnt+=1
    if d1_next!=d1:
        d1_cnt+=1
    if d2_next!=d2:
        d2_cnt+=1
    cnt_list = [a_cnt,b_cnt,c_cnt,d1_cnt,d2_cnt]
    return a_next, b_next, c_next, d1_next, d2_next, cnt_list

def metropolis_Hastings_solve(real_value,initial_value,var,N_size=6000,drop=1000,sim_print=False):
    '''
    metropolis_Hastings 구현하는 함수
    :param initial_value:
    aa,bb,cc,d1,d2를 튜플로 입력
    :param var:
    랜덤생성한 데이터 셋
    :param N_size:
    반복횟수
    :param drop:
    초기 시행은 버리는데 버릴 시점 ex)drop=1000, N_size=10000이면 1001번부터 10000번까지의 계산 결과를 활용하여 parameter의 평균을 구함
    :param sim_print:
    시뮬레이션을 프린트할지 여부를 출력
    :return:
    (aa,bb,cc,d1,d2)의 평균값
    '''
    delta = Decimal(2.4/sqrt(5))
    W = delta*Decimal(1e-3)
    aa,bb,cc,d1,d2,radius = [Decimal(i) for i in initial_value] # decimal 로 형변환
    raa,rbb,rcc = [Decimal(i) for i in real_value] # decimal 로 형변환
    hyper_param = set_parameter(raa,rbb,rcc)
    var = [Decimal(i) for i in var]
    param_dict ={"a_list":[aa],"b_list":[bb],"c_list":[cc],"d1_list":[d1],"d2_list":[d2]}
    cnt_list = [0,0,0,0,0]
    for epoch in range(N_size):
        if epoch%100==0:
            print('{}회 실행'.format(epoch))
        if sim_print==True:
            print("iter : {}\n aa의 현재 : {} 평균 : {}\n bb 현재 : {} 평균 : {}\n cc 현재 : {} 평균 : {}\n d1 현재 : {} 평균 : {}\n d2 현재 : {} 평균 : {}"\
                  .format(epoch,aa,np.mean(param_dict['a_list'])\
                          ,bb,np.mean(param_dict['b_list'])\
                          ,cc,np.mean(param_dict['c_list'])\
                          ,d1,np.mean(param_dict['d1_list'])\
                          ,d2,np.mean(param_dict['d2_list'])))
        # 파라메터 업데이트
        if epoch ==0:
            normal_set = aa,sqrt(1),bb,sqrt(1),cc,sqrt(1)
        else:
            a_mu = param_dict['a_list'][-1]
            a_var = sqrt(delta*np.var(param_dict['a_list'][:-1])+W)/10
            b_mu = param_dict['b_list'][-1]
            b_var = sqrt(delta*np.var(param_dict['b_list'][:-1])+W)/10
            c_mu = param_dict['c_list'][-1]
            c_var = sqrt(delta*np.var(param_dict['c_list'][:-1])+W)/10
            normal_set = (a_mu,a_var,b_mu,b_var,c_mu,c_var)
        aa,bb,cc,d1,d2,cnt_list = metropolis_Hastings(aa,bb,cc,d1,d2,var,radius,normal_set,hyper_param,cnt_list)

        # 파라메터 저장
        param_dict['a_list'].append(aa)
        param_dict['b_list'].append(bb)
        param_dict['c_list'].append(cc)
        param_dict['d1_list'].append(d1)
        param_dict['d2_list'].append(d2)

    # 결과 drop번째부터 계산
    ans = (np.mean(param_dict['a_list'][drop:]),np.mean(param_dict['b_list'][drop:]),np.mean(param_dict['c_list'][drop:]),np.mean(param_dict['d1_list'][drop:]),np.mean(param_dict['d2_list'][drop:]))
    return ans,param_dict,cnt_list

def metropolis_Hastings2(aa,bb,cc,d1,d2,var,radius,normal_set,hyper_param,cnt_list):
    a_cnt,b_cnt,c_cnt,d1_cnt,d2_cnt = cnt_list
    """
    MCMC 1회 연산하는 함수
    :param aa,bb,cc,d1,d2: 초기값
    :param var: 뽑은 데이터 셋

    :return: 연산이 끝난 후의 값
    """
    a_mu,a_var,b_mu,b_var,c_mu,c_var=normal_set
    a_var = 0.01
    a1,b1,a2,b2,a3,b3 = hyper_param
    N = len(var)  # 생성한 데이터의 개수

    # 업데이트 A
    a_tmp = Decimal(np.random.normal(a_mu,a_var))  # 정규 분포에서 임시값 뽑기
    while a_tmp<0:
        a_tmp = Decimal(np.random.normal(a_mu, 0.01))
    #print('a_tmp',a_tmp,aa,a_mu,a_var)
    ratio_a = Decimal(M(aa, bb, cc, d1, d2)[0] / M(a_tmp, bb, cc, d1, d2)[0]) ** N*(a_tmp/aa)**(a1-Decimal(1))*((sum(var)+b1)*(aa-a_tmp)).exp()  # r구하기
    #print('a prob',ratio_a)

    a_next = update(min(1,ratio_a), a_tmp, aa)


    # 업데이트 B
    b_tmp = Decimal(uniform(0,b_mu))
    while b_tmp<0:
        b_tmp = Decimal(np.random.normal(b_mu, b_var))
    #print('b_tmp',b_tmp,b_mu,b_var)
    ratio_b = Decimal(M(a_next, bb, cc, d1, d2)[0] / M(a_next, b_tmp, cc, d1, d2)[0]) ** N \
            * reduce(mul, [((-(b_tmp + cc * i) * d1).exp() - (-(b_tmp + cc * i) * d2).exp()) / (
            (-(bb + cc * i) * d1).exp() - (-(bb + cc * i) * d2).exp()) * (bb + cc * i) / (b_tmp + cc * i) for i in var])
    #print('b prob', ratio_b)
    b_next = update(min(1,ratio_b), b_tmp, bb)

    # 업데이트 C
    c_tmp = Decimal(uniform(0,c_mu))
    while c_tmp<0:
        c_tmp = Decimal(np.random.normal(c_mu, c_var))

    #print('c_tmp',c_tmp,c_mu,c_var)
    ratio_c = Decimal(M(a_next, b_next, cc, d1, d2)[0] / M(a_next, b_next, c_tmp, d1, d2)[0]) ** N \
            * reduce(mul, [((-(b_next + c_tmp * i) * d1).exp() - (-(b_next + c_tmp * i) * d2).exp()) / (
            (-(b_next + cc * i) * d1).exp() - (-(b_next + cc * i) * d2).exp()) * (b_next + cc * i) / (b_next + c_tmp * i) for i in var])
    #print('c prob', ratio_c)
    c_next = update(min(1,ratio_c), c_tmp, cc)

    d1_tmp = Decimal((1 + d2) / (1 + d2 * Decimal(1 - uniform())) - Decimal(1))
    while d1_tmp<0:
        d1_tmp = Decimal((1 + d2) / (1 + d2 * Decimal(1 - uniform())) - Decimal(1))  # 역함수 활용
    #print('d1_tmp',d1_tmp)
    ratio_d1 = Decimal(M(a_next, b_next, c_next, d1, d2)[0] / M(a_next, b_next, c_next, d1_tmp, d2)[0]) ** N \
            * reduce(mul, [((-(b_next + c_next * i) * d1_tmp).exp() - (-(b_next + c_next * i) * d2).exp()) / (
            (-(b_next + c_next * i) * d1).exp() - (-(b_next + c_next * i) * d2).exp()) for i in var])
    d1_next = update(min(1,ratio_d1), d1_tmp, d1)

    # 업데이트 d2
    d2_tmp = Decimal(uniform(d2 - radius, d2 + radius))  # 수정예정
    while d2_tmp>2 or d2_tmp<d1_next:
        d2_tmp = Decimal(uniform(d2 - radius, d2 + radius))  # 수정예정

    ratio_d2 = Decimal(M(a_next, b_next, c_next, d1_next, d2)[0] / M(a_next, b_next, c_next, d1_next, d2_tmp)[0]) ** N \
            * reduce(mul, [((-(b_next + c_next * i) * d1_next).exp() - (-(b_next + c_next * i) * d2_tmp).exp()) / (
            (-(b_next + c_next * i) * d1_next).exp() - (-(b_next + c_next * i) * d2).exp()) for i in var])
    d2_next = update(min(1,ratio_d2), d2_tmp, d2)

    #print('ratio a : {:.4f} ratio b : {:.4f} ratio c : {:.4f} ratio d1 : {:.4f} ratio d2 : {:.4f}'.format(ratio_a,ratio_b,ratio_c,ratio_d1,ratio_d2))
    if a_next!=aa:
        a_cnt+=1
    if b_next!=bb:
        b_cnt+=1
    if c_next!=cc:
        c_cnt+=1
    if d1_next!=d1:
        d1_cnt+=1
    if d2_next!=d2:
        d2_cnt+=1
    cnt_list = [a_cnt,b_cnt,c_cnt,d1_cnt,d2_cnt]
    return a_next, b_next, c_next, d1_next, d2_next, cnt_list

def metropolis_Hastings_solve2(real_value,initial_value,var,N_size=6000,drop=1000,sim_print=False):
    '''
    metropolis_Hastings 구현하는 함수
    :param initial_value:
    aa,bb,cc,d1,d2를 튜플로 입력
    :param var:
    랜덤생성한 데이터 셋
    :param N_size:
    반복횟수
    :param drop:
    초기 시행은 버리는데 버릴 시점 ex)drop=1000, N_size=10000이면 1001번부터 10000번까지의 계산 결과를 활용하여 parameter의 평균을 구함
    :param sim_print:
    시뮬레이션을 프린트할지 여부를 출력
    :return:
    (aa,bb,cc,d1,d2)의 평균값
    '''
    delta = Decimal(2.4/sqrt(5))
    W = delta*Decimal(1e-3)
    aa,bb,cc,d1,d2,radius = [Decimal(i) for i in initial_value] # decimal 로 형변환
    raa,rbb,rcc = [Decimal(i) for i in real_value] # decimal 로 형변환
    hyper_param = set_parameter(raa,rbb,rcc)
    var = [Decimal(i) for i in var]
    param_dict ={"a_list":[aa],"b_list":[bb],"c_list":[cc],"d1_list":[d1],"d2_list":[d2]}
    cnt_list = [0,0,0,0,0]
    for epoch in range(N_size):
        if epoch%100==0:
            print('{}회 실행'.format(epoch))
        if sim_print==True:
            print("iter : {}\n aa의 현재 : {} 평균 : {}\n bb 현재 : {} 평균 : {}\n cc 현재 : {} 평균 : {}\n d1 현재 : {} 평균 : {}\n d2 현재 : {} 평균 : {}"\
                  .format(epoch,aa,np.mean(param_dict['a_list'])\
                          ,bb,np.mean(param_dict['b_list'])\
                          ,cc,np.mean(param_dict['c_list'])\
                          ,d1,np.mean(param_dict['d1_list'])\
                          ,d2,np.mean(param_dict['d2_list'])))
        # 파라메터 업데이트
        if epoch ==0:
            normal_set = aa,sqrt(1),bb,sqrt(1),cc,sqrt(1)
        else:
            a_mu = param_dict['a_list'][-1]
            a_var = sqrt(delta*np.var(param_dict['a_list'][:-1])+W)/10
            b_mu = param_dict['b_list'][-1]
            b_var = sqrt(delta*np.var(param_dict['b_list'][:-1])+W)/10
            c_mu = param_dict['c_list'][-1]
            c_var = sqrt(delta*np.var(param_dict['c_list'][:-1])+W)/10
            normal_set = (a_mu,a_var,b_mu,b_var,c_mu,c_var)
        aa,bb,cc,d1,d2,cnt_list = metropolis_Hastings2(aa,bb,cc,d1,d2,var,radius,normal_set,hyper_param,cnt_list)

        # 파라메터 저장
        param_dict['a_list'].append(aa)
        param_dict['b_list'].append(bb)
        param_dict['c_list'].append(cc)
        param_dict['d1_list'].append(d1)
        param_dict['d2_list'].append(d2)

    # 결과 drop번째부터 계산
    ans = (np.mean(param_dict['a_list'][drop:]),np.mean(param_dict['b_list'][drop:]),np.mean(param_dict['c_list'][drop:]),np.mean(param_dict['d1_list'][drop:]),np.mean(param_dict['d2_list'][drop:]))
    return ans,param_dict,cnt_list
