''' This is a manual computation of the jacobian of a function
   demonstrating the use of the scan operation. Usually we 
   would just use the theano.gradient.jacobian method.
'''

import theano
import theano.tensor as T
x = T.dvector()
y = x ** 2
J, updates = theano.scan(lambda i, y, x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y,x])
f = theano.function([x], J, updates=updates)
print f([4,4])
