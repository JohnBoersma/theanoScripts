''' demonstration of (1d) gradient in theano with plot
'''

# pylint: disable = bad-whitespace, invalid-name, no-member

import numpy as np
import matplotlib.pyplot as plt
from theano import tensor as T
from theano import function as F

x = T.dscalar()
s = 1 / (1 + T.exp(-x))
gs = T.grad(s,x)
logistic = F([x],s)
dlogistic = F([x],gs)
plotrange = np.arange(-10,10,0.01)
plt.plot(plotrange,[logistic(i) for i in plotrange])
plt.annotate(r'$\sigma(x)$', color='b', xy=(1,0.5), xytext=(1.5,0.7),fontsize=20)
plt.annotate(r'$\frac{d\sigma(x)}{dx}$', color='r', xy=(1,0.5), xytext=(2.0,0.1),fontsize=20)
plt.plot(plotrange,[dlogistic(i) for i in plotrange],'r')
plt.show()
