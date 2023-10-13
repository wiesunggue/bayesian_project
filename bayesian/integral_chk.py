import sympy as sym
from sympy import Integral, symbols # cas 적분해주는 sympy 모듈 설치하려면 # pip install sympy를 cmd에 입력하면 된다
from math import exp, log10
from functools import reduce
# 적분 모듈(sympy)을 사용하는 법
# 1. symbols함수를 통해서 변수 선언      코드상 사용할 변수 = symbols('출력할 변수 이름')
# 2. Integral 함수에 넣어서 정적분을 하려면 적분 변수만 대입 Integral(수식, x).doit()
# 2. Integral 함수에 넣어서 부정적분을 하려면 적분 변수와 구간을 튜플로 대입 Integral(수식, (x, 0, sym.oo)).doit()
# 3. 함수 출력
w = symbols('w')
w1 = symbols('w1')
w2 = symbols('w2')
lam = symbols('lam')
x1 = symbols('x1')
x2 = symbols('x2')
### m1을 적분하는 함수, 이상 없음
m1_function = (w1+(1-w1)*sym.exp(-lam))*(w2+(1-w2)*sym.exp(-lam))*(1-w1)*(1-w2)*sym.exp(-2*lam)*lam**(x1+x2) # prior을 1로 하려면 -0.5를 없애면 된다
w1_inte_m1func =Integral(m1_function,(w1,0,1)).doit()
w2_inte_m1func =Integral(w1_inte_m1func,(w2,0,1)).doit()
m1 = Integral(w2_inte_m1func,(lam,0,sym.oo)).doit() # sym.oo 는 무한대
print(m1)

### m0를 적분하는 함수
m0_function = (w**2+2*w*(1-w)*sym.exp(-lam)+(1-w)**2*sym.exp(-2*lam))*(1-w)**2*sym.exp(-2*lam)*lam**(x1+x2)# prior을 1로 하려면 -0.5를 없애면 된다
w_inte_m0func = Integral(m0_function,(w,0,1)).doit()
m0 = Integral(w_inte_m0func,(lam,0,sym.oo)).doit() # sym.oo 는 무한대
print(m0)

x = symbols('x')

print(Integral(1/(x**10)*w,(x,1,sym.oo)).doit()) # 예제 코드 1/x^2의 적분

a = [0,0,[1,2],0]
