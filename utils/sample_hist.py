import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['axes.unicode_minus']=False # display '-'
data = np.random.randn(20)
print(data)
"""
data: data
bins: 長方形數目 (預設10)
normed: 將Y標準化
facecolor: 填滿顏色
edgecolor: 框框顏色
alpha: 透明度
"""
plt.hist(data, bins=40, normed=1, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel("Section")
plt.ylabel("P")
plt.title("Title")
plt.show()