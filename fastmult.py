''' This program fastmult.py explores
operations for multiplying matrices of
derivatives with vectors for performance
gains. This avoids actually calculating the
matrices of derivatives.
'''

import theano
import theano.tensor as T

''' First, let's clarify the Jacobian operation.
'''
W = T.dmatrix()
x = T.dvector()
y = T.dot(x,W)
J1 = theano.gradient.jacobian(y,x)
J2 = theano.gradient.jacobian(y,W)
f1 = theano.function([x,W], J1)
f2 = theano.function([x,W], J2)
print f1([0,1],[[1,1],[1,1]])
print f2([0,1],[[1,1],[1,1]])

''' The R-operator right-multiplies a Jacobian
by a vector.
'''

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
#print f([[1,1],[1,1]], [[2,2],[2,2]], [0,1])

''' That is, the Jacobian of y with respect to W is x^T, 
which is then right-multiplied by V.
'''
''' Similarly for the L-operator:
'''

v = T.dvector()
x = T.dvector()
y = T.dot(x,W)
VJ = T.Lop(y,W,v)
f = theano.function([v,x], VJ)
#print f([2,2], [0,1])
