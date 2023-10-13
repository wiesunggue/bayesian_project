import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from basis_stat import *

print(len(data),CV(data),skewness(data),kurtosis(data))
# 중심점은 다음과 같은 방법을 통해서 구함
iteration = 10**6
point = [0]*iteration
sd = [0]*iteration
for i in range(iteration):
        if i%(iteration//100)==0:
                print(f'{i} 회 진행')
        d_set = np.random.choice(data,size=200)
        point[i] = CV(d_set),skewness(d_set)
        sd[i] = np.mean(d_set)


def sig(mu, point):
    cnt = np.array([[0.0,0.0],[0.0,0.0]])
    for i in range(iteration):
        tmp = np.array([(point[i] - mu)])
        cnt += tmp.transpose().dot(tmp)
    return cnt / iteration

point = np.array(point)
print("표본 평균 : ",CV(data),sample_sd(data)**2,"부트스트랩 평균 : ",np.mean(sd,axis=0))
print("표본 평균 : ",np.mean(data),CV(data),skewness(data),"부트스트랩 평균 : ",np.mean(point,axis=0))
mu = np.mean(point,axis=0)
sigma = sig(mu,point)

print(mu,sigma)
hull = ConvexHull(point)


COV = np.array(sigma)

eigenvalues, eigenvectors = np.linalg.eig(COV)
print(COV,eigenvectors,eigenvalues)
theta = np.linspace(0, 2*np.pi, 1000);
ellipsis = 3*(np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))

for ax in (ax1, ax2):
    ax.plot(point[:, 0], point[:, 1], '.', color='k')
    if ax == ax1:
        ax.plot(CV(data),skewness(data) , 'o',color='b')
        ax.plot(CV(data) + ellipsis[0, :], skewness(data) + ellipsis[1, :])
        ax.set_title('Given points')
    else:
        ax.set_title('Convex hull')
        for simplex in hull.simplices:
            ax.plot(point[simplex, 0], point[simplex, 1], 'c')
        ax.plot(point[hull.vertices, 0], point[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
    ax.set_xlim([0.3,0.7])
    ax.set_ylim([0.3,1.3])
    ax.set_xticks(np.linspace(0.3,0.7,15))
    ax.set_yticks(np.linspace(0.3,1.3,15))
    ax.grid(True)
plt.savefig('simulcheck.png')
plt.show()