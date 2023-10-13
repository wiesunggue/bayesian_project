from hidden_function import *
import pickle
# 주의!!!
# 함수의 기울기를 통한 적분이기 때문에 너무 큰 구간을 탐색하면 대부분 0을 출력해버린다.
# hidden trucation 변수들
aa,bb,cc,d1,d2=0.2,0.3,0.4,0.1,1
aa, bb, cc, d1, d2 = [Decimal(i) for i in (aa,bb,cc,d1,d2)]  # decimal 로 형변환
# gamma function에 들어가는 상수 alpha beta
a1,b1=1,1
a2,b2=1,1
a3,b3=1,1

sample_size = 100
M = lambda aa,bb,cc,d1,d2:(Decimal(quad(lambda y : (-bb*Decimal(y)).exp()/(aa+cc*Decimal(y)),d1,d2)[0]),0)
mconst = (M(aa,bb,cc,d1,d2)[0]) # aa=2,bb=3,cc=4,d1=1,d2=10에서 0.002323084 R코드와 비교해서 잘 나오는지 확인 완료
# 내부 함수
# pdf함수 정의, x=0.1,aa=2,bb=3,cc=4,d1=1,d2=10조건에서 3.459362 확인 완료, 적분도 결과 확인 완료
dhtexp = lambda x,aa,bb,cc,d1,d2:(-aa*Decimal(x)).exp()*((-(bb+cc*Decimal(x))*d1).exp()-(-(bb+cc*Decimal(x))*d2).exp())/(mconst*(bb+cc*Decimal(x)))
# cdf함수 정의
phtexp = lambda x,aa,bb,cc,d1,d2:quad(dhtexp,0,x,args=(aa,bb,cc,d1,d2))
# cdf에서 역함수를 구하는 대신 qq에 해당하는 값을 구하는 함수 qq=0.4에서 0.07197확인 완료
# cdf이므로 근을 찾기 위해 좀더 효율적인 이분탐색 진행하였음
qhtexp = lambda qq,aa,bb,cc,d1,d2,imax=599:bisect(lambda x:(phtexp(x,aa,bb,cc,d1,d2)[0]-qq),0,imax)
#qhtexp = lambda qq,aa,bb,cc,d1,d2,imax=599:bisect(lambda x:(phtexp(x,aa,bb,cc,d1,d2)[0]-qq),0,imax)
# 0~1의 균일 분포에서 뽑은 랜덤한 값을 qhtexp에 넣은 값의 배열을 추출하는 과정
rhtexp = lambda N,aa,bb,cc,d1,d2,imax=599:([Decimal(qhtexp(uniform(),aa,bb,cc,d1,d2)) for _ in range(N)])

# data 500개면 약 데이터의 총 합이 70정도됨
start = time.time()
var = rhtexp(sample_size,aa,bb,cc,d1,d2)
#x=np.linspace(0.1,10,1000)
#x = [Decimal(i) for i in x]
#y=[dhtexp(i,aa,bb,cc,d1,d2) for i in x]
#plt.plot(x,y)
#plt.hist(var,bins=50,density=True) # 그림그리기
#plt.show()
end = time.time()
radius = 0.1
# 데이터 100 개 기준 11 초 정도 걸림
# 데이터 500 개 기준 54 초 정도 걸림(decimal 타입으로 소수점 30~50자리까지 계산)
# 데이터 1000개 기준 110초 정도 걸림...?
print("데이터 생성 시간 : {:.4f}(s)\n ".format(end-start),var[0])
print("데이터의 총 합 : ",sum(var))
initial_value=[1,1,1,0.5,1.5,radius]
real_value = (aa,bb,cc)
#a1, b1, a2, b2, a3, b3 = set_parameter(aa, bb, cc)

# prior 에 대한 그래프를 그려주는 부분 실행 안하려면 mode = 5
mode=5
if mode==1:
    AAAA = [P_A(i,bb,cc,d1,d2,var) for i in x]
    print('a finish')
    plt.plot(x, AAAA)
    plt.title('about A')
    plt.savefig('AAAA.png')
elif mode==2:
    BBBB = [P_B(aa,i,cc,d1,d2,var) for i in x]
    print('b finish')
    plt.plot(x, BBBB)
    plt.title('about B')
    plt.savefig('BBBB.png')
elif mode==3:
    CCCC = [P_C(aa,bb,i,d1,d2,var) for i in x]
    print('c finish')
    plt.plot(x, CCCC)
    plt.title('about C')
    plt.savefig('CCCC.png')
elif mode==4:
    DDDD = [P_d1(aa,bb,cc,i,d2,var) if i<1 else 0 for i in x]
    print('d finish')
    plt.plot(x,DDDD)
    plt.title('about d1')
    plt.savefig('DDDD.png')
#print(M(aa,bb,cc,d1,d2)[0]**500)
start= time.time()
ans,param_dict,cnt_list = metropolis_Hastings_solve(real_value,initial_value,var,N_size=5000,drop=100,sim_print=False)
#ans,param_dict,cnt_list = MCMC_solve(initial_value,var,N_size=1000,drop=100,sim_print=True)
end = time.time()
print('실행시간 : {}s '.format(end-start))
new_a,new_b,new_c,new_d1,new_d2 = ans
print(*ans,sep='\n')
print(cnt_list)
#print(param_dict)

# 로그 저장
with open('param_dict.pickle','wb') as fw:
    pickle.dump(param_dict,fw)
plt.plot(param_dict['a_list'])
plt.show()
# 랜덤 추출 시뮬레이터
#a=time.time()
#rx = rhtexp(50,aa,bb,cc,d1,d2)
#b=time.time()
#print("실행시간 : ",b-a)
#seq = np.linspace(0,1.2,100)
#plt.hist(rx,bins=40,density=True)
#plt.plot(seq,np.array([dhtexp(i,aa,bb,cc,d1,d2) for i in seq]))
#plt.show()

# 일단 다 됐는데 피드백 받아야 할 것
# delta = 2
# k = dimension이라서 aa,bb,cc,d1,d2로 5를 선택했는데 5가 맞는지
# 초기 분산을 1 초기 평균을 aa,bb,cc로 뒀는데 이러면 될지
# 입실론 값 1 을 어떻게 정할지

