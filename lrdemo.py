'''  program lrdemo.py
  This program demonstrates a solution to a toy logistic regression problem
  using gradient descent in theano.
  Convergence confirmed for w = +- 20 (with b=0), b = +- 10 (with w = 1)
  Convergence problems can occur - see
  http://www2.sas.com/proceedings/forum2008/360-2008.pdf
'''

# pylint: disable = bad-whitespace, invalid-name, no-member

import argparse
import numpy as np
import matplotlib.pyplot as plt
from theano import tensor as T
from theano import shared as S
from theano import function as F


def f(ai,bi,t):
    ''' sigmoid function with arbitrary slope ai
	    and offset bi
    '''
    return 1 / (1 + np.exp(-ai*t-bi))

def invf(ai,bi,t):
    ''' inverse of sigmoid function to help define
        plot range
    '''
    return (np.log(t/(1-t)) - bi) / ai

# optional parameters to specify w and b of the target function
# and number of iterations

parser = argparse.ArgumentParser()
parser.add_argument('-w', type = float, default = 0.2, dest='w')
parser.add_argument('-b', type = float, default = 0.0, dest='b')
parser.add_argument('-N', type = int, default = 1000, dest='N')
wt = parser.parse_args().w
bt = parser.parse_args().b
iterations = parser.parse_args().N

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
y = T.vector()
sig = 1 / (1 + T.exp(-T.dot(X, W) - B))
xent = -y * T.log(sig) - (1-y) * T.log(1 - sig)
cost = xent.mean()
gw, gb = T.grad(cost, [W, B])

# compile theano functions

TRAIN = F(
    inputs=[X, y],
    outputs=[W, B],
    updates=((W, W - 0.1 * gw), (B, B - 0.1 * gb)))

# train model

for i in range(iterations):
    W, B = TRAIN(xtarg, ytarg.flatten())

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
