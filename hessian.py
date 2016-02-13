''' This program hessian.py demonstrates
calculation of a Hessian matrix in theano.

The Hessian is defined as the matrix of
second order partial derivatives of a
scalar function of a vector.
'''
# pylint: disable = bad-whitespace, invalid-name, no-member, bad-continuation

import theano
import theano.tensor as T

x = T.dvector()
y = x **2
cost = y.sum()
gy = T.grad(cost, x)
H = theano.scan(lambda i, gy, x : T.grad(gy[i],x), sequences=T.arange(gy.shape[0]),
                         non_sequences=[gy, x])[0]
f = theano.function([x], H)
print f([8,82])
