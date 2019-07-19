from peakdetect.peakdetect import peakdetect
import numpy as np
from matplotlib import pyplot as plt

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
plt.plot(result)
plt.plot(peaks, result[peaks], "x")
plt.plot(np.zeros_like(result), "--", color="gray")
ls = np.arange(gtruth.shape[0])
plt.scatter(ls, gtruth)
plt.plot(ls, gtruth)
plt.xlabel('Frame')
plt.ylabel('Probability of Changing Point Frame')
plt.ylim([-0.1, 1.1])
plt.show()