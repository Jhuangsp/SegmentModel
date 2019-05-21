
import math
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import special

plt.rcParams.update({'font.size': 15})

def discrete_gaussian_kernel(t, n):
    return math.exp(-t) * special.iv(n, t)
'''
How to use discrete_gaussian_kernel()
scipy.special.iv(n, t): Modified Bessel function of the first kind of real order.
'''
ns = np.arange(-5, 5+1)
y0 = discrete_gaussian_kernel(0.5, ns)
y1 = discrete_gaussian_kernel(1, ns)
y2 = discrete_gaussian_kernel(2, ns)
plt.plot(ns, y0, ns, y1, ns, y2)
plt.xlabel('Frame')
plt.ylabel('Weight')
plt.xlim([-5, 5])
plt.ylim([0, 1])
plt.show()