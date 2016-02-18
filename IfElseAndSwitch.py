''' This program demonstrates conditional control
    in theano.
'''

# pylint: disable = bad-whitespace, invalid-name, no-member, bad-continuation

import time
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

# first, let's explore T.mean()

x = T.matrix()
mtest = T.mean(x)
ftest = theano.function([x], mtest)
mat = [[1,2],[3,4]]
print ftest(mat)

# Now for conditional operators
# This kind of mutliple assignment requires argument

a,b = T.scalars('a','b')
x,y = T.matrices('x','y')

# switch takes a tensor as a condition and two input variables
# switch is elementwise so more general
# ifelse takes a boolean condition and two variables
# ifelse evaluates only one output (with cvm linker) so can be faster
# documentation says cvm is not yet the default but apparently it is now

z_switch = T.switch(T.lt(a,b), T.mean(x), T.mean(y))
z_lazy = ifelse(T.lt(a,b), T.mean(x), T.mean(y))

# vm linker is in t tutorial but does not seem to be described in documentation
# but cvm is documented and seems a bit faster - using that.

f_switch = theano.function([a,b,x,y], z_switch, mode=theano.Mode(linker='cvm'))
f_lazyifelse = theano.function([a,b,x,y], z_lazy)

val1 = 0.
val2 = 1.
big_mat1 = np.ones((10000,1000))
big_mat2 = np.ones((10000,1000))

n_times = 10

# time.clock() is active cpu time

tic = time.clock()
for i in range(n_times):
    f_switch(val1,val2,big_mat1,big_mat2)

print 'time spent evaluating both values %f sec' % (time.clock() - tic)

tic = time.clock()
for i in range(n_times):
    f_lazyifelse(val1,val2,big_mat1,big_mat2)
print 'time spend evaluating one value %f sec'  %  (time.clock() - tic)
