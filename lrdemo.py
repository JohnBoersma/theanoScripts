'''  program lrdemo.py
  this program demonstrates a tiny solution to a logistic regression problem
  using gradient descent in theano
'''

import numpy as np
import matplotlib.pyplot as plt
from theano import tensor as T
from theano import shared as S
from theano import function as F

#  training data
#  the double brackets below shape the array in numpy so not zero height
X0 = np.array([[-0.15, 0.1, 0.2, 0.12]])
Y0 = np.array([0, 0, 1, 1])
W = S(1.2)
B = S(0.0)

X = T.matrix()
Y = T.vector()
SIG = 1 / (1 + T.exp(-T.dot(X, W) - B))
PREDICTION = SIG > 0.5
XENT = -Y * T.log(SIG) - (1-Y) * T.log(1 - SIG)
COST = XENT.mean()
GW, GB = T.grad(COST, [W, B])

# compile

# when you call theano.function, the first parameter should be a list of
# symbolic variables, not values. Then, you need to call the returned
# object, f, on values, not symbolic variables.

TRAIN = F(
    inputs=[X, Y],
    outputs=[PREDICTION, XENT],
    updates=((W, W - 0.1 * GW), (B, B - 0.1 * GB)))
PREDICT = F(inputs=[X], outputs=PREDICTION)

#  10k iterations doesn't get it, 100k does

for i in range(10000):
    pred, err = TRAIN(X0, Y0)
print Y0
print PREDICT(X0)

def f(a,b,t):
	return 1 / (1 + np.exp(-a*t+b))

t1 = np.arange(-40.0, 40.0, 0.1)
t2 = np.arange(-40.0, 40.0, 1.0)
a = 1
b = 0
plt.plot(t1, f(a,b,t1), 'b', linewidth=2)
a = 0.2
b = 2
plt.plot(t2, f(a,b,t2), 'bo')
plt.plot()
plt.show()