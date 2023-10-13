from generater import correction_F,ZIP
import matplotlib.pyplot as plt

import sys
rinput = sys.stdin.readline
N=int(input())
a={}
for i in range(N):
    k = int(rinput())
    if a.get(k)==None:
        a[k]=1
    else:
        a[k]+=1

m=[0,0]
for idx,i in sorted(a.items()):
    if i>m[1]:
        m = idx,i

print(m[0])