
import math
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import special

plt.rcParams.update({'font.size': 20})

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
    # normalized = normalized/unit
    return normalized
    # return normalized.reshape(-1)


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
    # plt.plot(ls, weighted_data_list)

    # plt.xlabel('Frame')
    # plt.ylabel('Probability of Changing Point Frame')
    # plt.xlim([30, 70])
    # plt.ylim([-0.1, 1.1])
    # plt.show()
    return weighted_data_list

def gaussian_like_weighted(data_list):
    '''
    Apply different size gaussian mask to impulse frame according to 
    different output steps.
    Parameter:
     - target_label: all steps of output label (shape: decoder_steps, output length)
    Return:
     - different size gaussian weighted target_label
    '''    
    ns = np.arange(11)-5
    w = discrete_gaussian_kernel(2, ns)*3
    weighted_data_list = np.convolve(data_list, w, 'same')

    # ls = np.arange(len(weighted_data_list))
    # plt.scatter(ls, weighted_data_list)
    # plt.plot(ls, weighted_data_list)

    # plt.xlabel('Frame')
    # plt.ylabel('Probability of Changing Point Frame')
    # plt.xlim([30, 70])
    # plt.ylim([-0.1, 1.1])
    # plt.show()
    return weighted_data_list


def oblique_mean(data):
    rows, cols = data.shape
    ans     = np.zeros((rows+cols-1), dtype=np.float32)
    divider = np.zeros((rows+cols-1), dtype=np.float32)
    tmp = np.ones((cols), dtype=np.float32)
    for i in range(rows):
        ans[i:i+cols] += data[i,:]
        divider[i:i+cols] += tmp
    return (ans/divider)



def draw(args, result, gt):
    result_name = "./model/result.npy"
    gt_name = "./model/gt.npy"

    ls = np.arange(result.shape[0]) + (args.in_frames - args.out_band) // 2
    plt.scatter(ls, result, color='orange')
    plt.plot(ls, result, color='orange')
    np.save(result_name, result)

    ls = np.arange(len(gt))
    plt.scatter(ls, gt)
    plt.plot(ls, gt)
    np.save(gt_name, gt)

    plt.xlabel('Frame')
    plt.ylabel('Probability of Changing Point Frame')
    plt.ylim([-0.1, 1.1])
    plt.show()