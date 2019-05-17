
import math
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import special

def normalize(step_input):
    '''
    Normaliaze the joint coordinate which take Neck(idx:1) as origin, 
    distance from Neck(idx:1) to Nose(idx:0) as length unit.
    Parameter:
     - step_input: input of one step (shape: input_size)
    Return:
     - normalized step_input
    '''
    step_input = step_input.reshape(-1,3)
    normalized = step_input[:,:2]
    normalized = normalized - normalized[1,:]
    unit = np.linalg.norm(normalized[0])
    normalized = normalized/unit
    step_input[:,:2] = normalized
    return step_input.reshape(-1)


def discrete_gaussian_kernel(t, n):
    return math.exp(-t) * special.iv(n, t)
'''
How to use discrete_gaussian_kernel()
scipy.special.iv(n, t): Modified Bessel function of the first kind of real order.

ns = np.arange(-5, 5+1)
y0 = discrete_gaussian_kernel(0.5, ns)
y1 = discrete_gaussian_kernel(1, ns)
y2 = discrete_gaussian_kernel(2, ns)
plt.plot(ns, y0, ns, y1, ns, y2)
plt.xlim([-4, 4])
plt.ylim([0, 0.7])
plt.show()
'''

def gaussian_weighted(data_list, n):
    '''
    Apply different size gaussian mask to impulse frame according to 
    different output steps.
    Parameter:
     - target_label: all steps of output label (shape: decoder_steps, output length)
    Return:
     - different size gaussian weighted target_label
    '''
    ns = np.arange(11)-5
    w = discrete_gaussian_kernel(n, ns)
    weighted_data_list = np.convolve(data_list, w, 'same')

    # ls = np.arange(len(weighted_data_list))
    # plt.scatter(ls, weighted_data_list)
    # plt.show()
    return weighted_data_list