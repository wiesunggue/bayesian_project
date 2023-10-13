import numpy as np
from matplotlib import pyplot as plt
from math import pi
from basis_stat import *

u=CV(data)     #x-position of the center
v=skewness(data)    #y-position of the center
print(u,v)
a=1     #radius on the x-axis
b=1    #radius on the y-axis

t = np.linspace(0, 2*pi, 100)
plt.plot( u+a*np.cos(t) , v+b*np.sin(t) )
plt.grid(color='lightgray',linestyle='--')
plt.show()