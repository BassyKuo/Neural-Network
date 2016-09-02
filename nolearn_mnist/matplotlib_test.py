import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,5,0.01)
y = np.sin(x*5)
plt.plot(x,y)
plt.title(s='title')
plt.xlabel(s='x_value',fontsize='small',verticalalignment='top',horizontalalignment='center')
plt.ylabel(s='sin(x)',fontsize='small',verticalalignment='top',horizontalalignment='center')
plt.savefig('test.png')
plt.show()

plt.plot(x,y/5)
plt.show()
############################################################################
plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])

plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.figure(1)                # figure 1 current; subplot(212) still current
plt.subplot(211)             # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3') # subplot 211 title
plt.show()
