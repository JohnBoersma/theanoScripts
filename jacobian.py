''' This is a manual computation of the jacobian of a function
   demonstrating the use of the scan method. Usually we 
   would just use the theano.gradient.jacobian method.
'''

import theano
import theano.tensor as T
x = T.dvector()
y = x ** 2

# if, instead of a fixed number of iterations, we want to iterate over an object,
# that object is specified as a 'sequence'
# constants in a scan are 'non_sequences'
# .shape returns and Ivector with the shape of the tensor
# so y.shape[0] is the  size of y along the first axis
# a single element has shape 1, not 0
# arange starts at 0 index and ends at the integer before stop
# so [1,2,3] has shape 3 and  arange (0,1,2)

J = theano.scan(lambda i, y, x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y,x])[0]
f = theano.function([x], J)
print f([2,2])
