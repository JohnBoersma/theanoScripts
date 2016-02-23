''' Examples of shape inference'''

import numpy as np
import theano
import theano.tensor as T

# pylint: disable = bad-whitespace, invalid-name, no-member, bad-continuation, assignment-from-no-return

# here the debug name is needed - used in debugprint

x = T.matrix('x')
f = theano.function([x], (x ** 2).shape)

# print a computation graph

theano.printing.debugprint(f)

# shape inference problem

x = T.matrix('x')
y = T.matrix('y')

# Hmm... join does not seem to be documented anywhere

z = T.join(0, x, y)

# a uniform distribution over 0,1 in a 5x4 tensor

xv = np.random.rand(5,4)
yv = np.random.rand(3,3)

f = theano.function([x,y], z.shape)
theano.printing.debugprint(f)

# should lead to error of mismatched indices but does not

print f(xv, yv)

# instead, compute values and not just shape
# and an error is thrown

f = theano.function([x,y], z)
theano.printing.debugprint(f)
#print f(xv,yv)

# specifiying exact shape

x = T.matrix('x')
x_specify_shape = T.specify_shape(x, (2,2))
f = theano.function([x], (x_specify_shape ** 2).shape)
theano.printing.debugprint(f)
