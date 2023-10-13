import numpy as np

#try({m1[r]=integrate(Vectorize(function(ld){  ## prior = ld^(-0.5)
#((om.0+(1-om.0) * exp(-ld)) ^ alpha) * ((1-om.0) ^ (n-alpha)) * (ld ^ (t-1 / 2)) * exp(-ld * (n-alpha))
#}), 0, Sub.inf2)$value}, silent = TRUE)

om=0.5
size=100000
dist = np.arange(0,500,1/size)
def aaa(x):
    print(x**(108.5))
    return (0.5+0.5*np.exp(-x))**13*(0.5**37)*x**(108.5)*np.exp(-x*37)

print(sum(aaa(dist))/size)

def func(x):
    return x**2

d = np.arange(0,100,1/size)
print(sum(func(d))/size)
print(2**10*5)