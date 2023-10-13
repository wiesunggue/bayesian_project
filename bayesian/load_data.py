import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma
plt.rcParams['font.family'] = 'Malgun Gothic'
with open('param_dict.pickle','rb') as fr:
    param_dict = pickle.load(fr)

# mode=1 aa의 변화
# mode=2 bb의 변화
# mode=3 cc의 변화
# mode=4 d1의 변화
# mode=5 d2의 변화
mode=2
#print(param_dict['c_list'])
if mode==1:
    plt.plot(param_dict['a_list'])
    plt.title("aa의 변화")
    plt.savefig('variation_of_aa.png')
elif mode==2:
    plt.plot(param_dict['b_list'])
    plt.title("bb의 변화")
    plt.savefig('variation_of_bb.png')
elif mode==3:
    plt.plot(param_dict['c_list'])
    plt.title("cc의 변화")
    plt.savefig('variation_of_cc.png')
elif mode==4:
    plt.plot(param_dict['d1_list'])
    plt.title("d1의 변화")
    plt.savefig('variation_of_d1.png')
elif mode==5:
    plt.plot(param_dict['d2_list'])
    plt.title("d2의 변화")
    plt.savefig('variation_of_d2.png')
elif mode==6:
    plt.plot(param_dict['a_list'])
    plt.plot(param_dict['b_list'])
    plt.plot(param_dict['c_list'])
    plt.plot(param_dict['d1_list'])
    plt.plot(param_dict['d2_list'])
    plt.title("모든 그림")
    plt.savefig('all_of_plot.png')
elif mode==7:
    x = np.linspace(0,10,100000)
    y = gamma(0.1,0,2).pdf(x)
    plt.plot(x,y)
#plt.show()
for i in param_dict:
    print(np.mean(param_dict[i][1000:]))