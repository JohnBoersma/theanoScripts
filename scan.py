''' This program explores the operation of the scan method in
    theano, following
    http://nbviewer.jupyter.org/gist/triangleinequality/1350873eebea33973e41

    scan is theano's native iterator, more efficient than external Python iteration
    can iterate over loops, over a tensor, elements of a vector, etc
'''

# pylint: disable = bad-whitespace, invalid-name, no-member

import numpy as np
import theano as th
import theano.tensor as T

# first, we count to 10 theano style

# i is the iterator
i = T.iscalar()

# the first argument of scan is the function which is applied on each iteration
# the function acts on the output of that function on the last iteration
# outputs_info initializes the value

# scan returns the tuple (outputs,updates). Either we accept them both, or slice off
# what we want. Scan appends to the list of results on each iteration.

results = th.scan(lambda previous_count: previous_count+1, outputs_info=0, n_steps=i)[0]

f = th.function(inputs=[i], outputs=results)
print f(17)

# if we just want the final result, this is better since it is more memory efficient

f1 = th.function(inputs=[i], outputs=results[-1])
print f1(17)

# but by storing interim results, we can calculate any kind of recurrance relation
# such as a Fibonacci sequence

i = T.iscalar()
x0 = T.ivector()

# now outputs_info is a list containing a dictionary.
# the initial varible is vector x0, and
# taps is the positions to read: the last two

results = th.scan(lambda f_m_1, f_m_2 : f_m_1 + f_m_2,
	                 outputs_info=[{'initial':x0, 'taps':[-2,-1]}], n_steps=i)[0]

f = th.function(inputs=[i,x0], outputs=results)

# need to cast as 32 bit integers to conform to ivector
# note that 50 iterations leads to overflow
# with symptom negative integers

print f(50, np.asarray([0,1], dtype=np.int32))

# to address the overflow, we can create
# a condition to scan
# which terminates the scan when
# it evaluates to True

def fib(f_m_1, f_m_2):
    ''' demonstrate a function used in scan
       with a termination condition
    '''
    ret = f_m_1 + f_m_2
    return ret, th.scan_module.until(ret < 0)

results = th.scan(fib, outputs_info=[{'initial':x0,'taps':[-2,-1]}],n_steps=i)[0]

f = th.function(inputs=[i,x0], outputs=results)
print f(50, np.asarray([0,1], dtype=np.int32))




