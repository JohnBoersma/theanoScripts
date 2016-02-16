''' This program fastmult.py explores
operations for multiplying matrices of
derivatives with vectors for performance
gains. This avoids actually calculating the
matrices of derivatives.
'''

# pylint: disable = bad-whitespace, invalid-name, no-member, bad-continuation

import theano
import theano.tensor as T

# First, let's clarify the Jacobian operation.

W = T.dmatrix()
x = T.dvector()
y = T.dot(x,W)
J1 = theano.gradient.jacobian(y,x)
J2 = theano.gradient.jacobian(y,W)
f1 = theano.function([x,W], J1)
f2 = theano.function([x,W], J2)
print f1([0,1],[[1,1],[1,1]])
print f2([0,1],[[1,1],[1,1]])

# The R-operator right-multiplies a Jacobian by a vector.

W = T.dmatrix()
V = T.dmatrix()
x = T.dvector()

''' Note that many but not all ops
can be embedded in ('support') an R-op.
dot can.

Arguments are:
Rop(function, derivatives with respect to, right-multiply)
'''

y = T.dot(x,W)
JV = T.Rop(y,W,V)
f = theano.function([W,V,x], JV)
print f([[1,1],[1,1]], [[2,2],[3,17]], [0,1])

# That is, the Jacobian of y with respect to W is x^T,
# which is then right-multiplied by V.
# For Rop, the first indices of grad multiplies V.
# Similarly for the L-operator:

v = T.dvector()
x = T.dvector()
y = T.dot(x,W)
VJ = T.Lop(y,W,v)
f = theano.function([v,x], VJ)
print f([2,2], [0,1])

# Note that no explicit value of W is provided here - it's not needed
# because the partial derivatives remove it. Grad works symbolically.
# But for some reason this doesn't work for Rop.
# I've asked on stack overflow what the deal is.
# For Lop, the last indices of grad multiplies V.

# Here's the first way to multiply a Hessian times a vector

x = T.dvector()
v = T.dvector()
y = T.sum(x ** 2)
gy = T.grad(y,x)
vH = T.grad(T.sum(gy * v), x)
f = theano.function([x,v], vH)
print f([4,4], [2,2])

# Or, can do the same using Rop:

x = T.dvector()
v = T.dvector()
y = T.sum(x ** 2)
gy = T.grad(y,x)
Hv = T.Rop(gy, x, v)
f = theano.function([x,v], Hv)
print f([4,4], [2,2])

# This works because of the symmetry of the Hessian matrix.
# Which is faster in a specific case should be determine
# by profilling.
