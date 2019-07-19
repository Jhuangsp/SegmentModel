from peakdetect.peakdetect import peakdetect
import numpy as np
from functools import reduce
import os
from matplotlib import pyplot as plt

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def post_process(data, it=5):
    # expand
    for i in range(it):
        data = reduce(np.union1d, (data, data-1, data+1))
    data = data[data>=0]

    groups = group_consecutives(data)

    return np.array([ g[len(g)//2] for g in groups ])


rtname = ".././model/result.npy"
gtname = ".././model/gt.npy"
result = np.load(rtname)
gtruth = np.load(gtname)
# result = np.pad(result, (5, 5), 'edge')


# peaks = np.array(peakdetect(result, lookahead=28))

# ls = np.arange(result.shape[0])
# plt.scatter(ls, result, color='orange')
# plt.plot(ls, result, color='orange')

# ls = np.arange(gtruth.shape[0])
# plt.scatter(ls, gtruth)
# plt.plot(ls, gtruth)

# plt.scatter(peaks[0][:,0], peaks[0][:,1], color='red')
# # plt.scatter(peaks[1][:,0], peaks[1][:,1], color='red')

# plt.xlabel('Frame')
# plt.ylabel('Probability of Changing Point Frame')
# plt.ylim([-0.1, 1.1])
# plt.show()

from scipy.signal import find_peaks
peaks, _ = find_peaks(result, height=0.05)
np.save('.././model/peaks.npy', peaks)

peaks = post_process(peaks, it=4)

ls = np.arange(gtruth.shape[0])
plt.scatter(ls, gtruth)
plt.plot(ls, gtruth)

plt.plot(result, color="orange")
plt.plot(peaks, result[peaks], "x", color="red")

for p in peaks:
    plt.plot((p,p), (0,1), "--", color="red")

plt.plot(np.zeros_like(result), "--", color="gray")
plt.xlabel('Frame')
plt.ylabel('Probability of Changing Point Frame')
plt.ylim([-0.1, 1.1])
plt.show()