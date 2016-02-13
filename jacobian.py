''' This is a manual computation of the Jacobian of a function
   demonstrating the use of the scan method. Usually we
   would just use the theano.gradient.jacobian method.
'''

# pylint: disable = bad-whitespace, invalid-name, no-member

import theano
import theano.tensor as T

x = T.dvector()
y = x ** 2

'''
 If, instead of a fixed number of iterations, we want to iterate over an object,
 that object is specified as a 'sequence'.

 Constants in a scan are 'non_sequences'.

 .shape returns an Ivector with the shape of the tensor,
 so y.shape[0] is the  size of y along the first axis.

 arange starts at 0 index and ends at the integer before stop,
 so [1,2,3] has shape 3 and  arange (0,1,2).

 this produces a Jacobian of the partial derivative of each component of y
 with respect to each component of x.

 Note that the non_sequences contribute x and y to the lambda function
 and sequences contributes the i; scan extracts the function parameters in order
 from the subsequent scan parameters.
'''

J = theano.scan(lambda i, y, x : T.grad(y[i], x), sequences=T.arange(y.shape[0]),
                        non_sequences=[y,x])[0]
f = theano.function([x], J)
print f([4,4])
