import matplotlib.pyplot as plt
import numpy as np
import matplotlib

a = np.arange(10)

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))
plt.rc('grid', linestyle="dotted", color='gray')
plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.scatter(a,a)
plt.grid(True)
plt.show()