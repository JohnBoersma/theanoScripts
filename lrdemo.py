'''  program lrdemo.py
  This program demonstrates a solution to a toy logistic regression problem
  using gradient descent in theano.
  Convergence confirmed for w = +- 20 (with b=0), b = +- 10 (with w = 1) 
  Convergence problems can occur - see
  http://www2.sas.com/proceedings/forum2008/360-2008.pdf
'''

# Questions: are we comparing sigmoids or binary [1,0]?
# Can cross-entropy be replaced with MLL?

import argparse
import numpy as np
import matplotlib.pyplot as plt
from theano import tensor as T
from theano import shared as S
from theano import function as F

# sigmoid function with arbitrary slope w and offset b

def f(a,b,t):
	return 1 / (1 + np.exp(-a*t-b))

# inverse of sigmoid function to help define plot range

def invf(a,b,t):
	return (np.log(t/(1-t)) - b) / a

# optional parameters to specify w and b of the target function
# and number of iterations

parser = argparse.ArgumentParser()
parser.add_argument('-w', type = float, default = 0.2, dest='w')
parser.add_argument('-b', type = float, default = 0.0, dest='b')
parser.add_argument('-iter', type = int, default = 1000, dest='iter')
wt = parser.parse_args().w
bt = parser.parse_args().b
iter = parser.parse_args().iter

#  generate training data set of four points equispaced in y

ytarg = np.array([[1./8,3./8,5./8,7./8]])
xtarg = invf(wt,bt,ytarg)

# initial values for model sigmoid

a = 1.0
b = 0.0
W = S(a)
B = S(b)

# symbolic computations for theano

X = T.matrix()
Y = T.vector()
SIG = 1 / (1 + T.exp(-T.dot(X, W) - B))
PREDICTION = SIG > 0.5
XENT = -Y * T.log(SIG) - (1-Y) * T.log(1 - SIG)
COST = XENT.mean()
GW, GB = T.grad(COST, [W, B])

# compile theano functions

TRAIN = F(
    inputs=[X, Y],
    outputs=[PREDICTION, XENT, W, B],
    updates=((W, W - 0.1 * GW), (B, B - 0.1 * GB)))
PREDICT = F(inputs=[X], outputs=PREDICTION)

# train model

for i in range(iter):
    pred, err, W, B = TRAIN(xtarg, ytarg.flatten())

# compare fitted results to intial model
# why does a perfect fit not plot that way for nonzero b?

print "a " + str(a)
print "W " + str(W)
print "b " + str(b)
print "B " + str(B)

plotrange = np.arange(np.amin(xtarg) * 1.1,np.amax(xtarg) * 1.1,0.01)
plt.plot(plotrange, f(a,b,plotrange), 'b', linewidth=2)
plt.plot(plotrange, f(W,B,plotrange), 'r', linewidth=2)
plt.plot(xtarg, ytarg, 'bo')
plt.show()
