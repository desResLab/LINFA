import numpy as np
import matplotlib.pyplot as plt
import itertools

fs=14

T = 350.0
P = 1.0

p0 = 1000.0
e0 = -21000.0

variable_inputs = [[350.0, 375.0 ,400.0, 425.0, 450.0],
                   [1.0, 2.0, 3.0, 4.0, 5.0]]

tp_pairs = list(itertools.product(*variable_inputs))

e_const = np.linspace(-30000,-15000,1000)

plt.figure(figsize=(5,3))
for t,p in tp_pairs:
	k_0 = (1/p0)*np.exp(-e0/(8.314*t))
	p_const = (1/k_0)*np.exp(-e_const/(8.314*t))
	plt.plot(p_const,e_const,label='t: '+str(t)+', p:'+str(p))
plt.scatter(p0,e0)
#plt.legend()
plt.xlim([500,1500])
plt.ylim([-23000,-18000])
plt.xlabel(r'$p_{0}$',fontsize=fs)
plt.ylabel(r'$e$')
plt.tight_layout()
plt.show()

