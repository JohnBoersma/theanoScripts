''' demonstration of (1d) gradient in theano
'''

# pylint: disable = bad-whitespace, invalid-name, no-member

import numpy as np
import matplotlib.pyplot as plt
from theano import tensor as T
from theano import function as F
from theano import shared as S

x = T.dscalar()
s = 1 / (1 + T.exp(-x))
gs = T.grad(s,x)
logistic = F([x],s)
dlogistic = F([x],gs)
plt.plot(np.arange(-10,10,0.01),[logistic(i) for i in np.arange(-10,10,0.01)])
plt.plot(np.arange(-10,10,0.01),[dlogistic(i) for i in np.arange(-10,10,0.01)],'r')
plt.show()