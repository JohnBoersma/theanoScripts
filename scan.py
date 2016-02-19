''' This program explores the operation of the scan method in
    theano, following
    http://nbviewer.jupyter.org/gist/triangleinequality/1350873eebea33973e41
    and the 'loop' section of the theano tutorial

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

print '\ntanh(x(t).dot(W) + b) elementwise\n'

X = T.matrix()
W = T.matrix()
# sym for symbolic
b_sym = T.vector()

# scan iterates over the first index of sequences
# so here, we iterate over a 2D matrix to get vectors

results = th.scan(lambda v: T.tanh(T.dot(v,W) + b_sym), sequences=X)[0]
compute_elementwise = th.function([X,W,b_sym], results)

x = np.eye(2, dtype=th.config.floatX)
w = np.ones((2,2), dtype=th.config.floatX)
b = np.ones((2), dtype=th.config.floatX)

print (compute_elementwise(x,w,b))

#comparison with numpy

print (np.tanh(x.dot(w) + b))

print '\nx(t) = tanh(x(t-1).dot(W) + y(t).dot(U) + p(T-t).dot(V))\n'

X = T.vector()
W = T.matrix()
b_sym = T.vector()
U = T.matrix()
Y = T.matrix()
V = T.matrix()
P = T.matrix()

# tm1 is t minus 1
# that is, the initialization state the first time
# and the output each other time

# [::-1] is an example of Python "step" or "stride" slicing. 
#  with -1, his reverses the order of a list
# scanning over two variables means 
# scanning them synchronously
# outputs info is the initial state on iteration
results = th.scan(lambda y,p,x_tm1 : T.tanh(T.dot(x_tm1,W) + T.dot(y,U) + T.dot(p,V)),
                          sequences = [Y,P[::-1]], outputs_info=[X])[0]
compute_seq = th.function([X,W,Y,U,P,V], results)

x = np.zeros((2), dtype=th.config.floatX)
x[1] = 1
w = np.ones((2,2), dtype=th.config.floatX)
y = np.ones((5,2), dtype = th.config.floatX)
y[0,:] = -3
u = np.ones((2,2), dtype = th.config.floatX)
p = np.ones((5,2), dtype = th.config.floatX)
p[0,:] = 3
v = np.ones((2,2), dtype = th.config.floatX)

print compute_seq(x,w,y,u,p,v)

# comparison with numpy

x_res = np.zeros((5,2), dtype=th.config.floatX)
x_res[0] = np.tanh(x.dot(w) + y[0].dot(u) + p[4].dot(v))
for i in range(1,5):
    x_res[i] = np.tanh(x_res[i-1].dot(w) + y[i].dot(u) + p[4-i].dot(v))
print x_res

# computing the norms of rows of X

X = T.matrix()
results = th.scan(lambda x_i : T.sqrt((x_i ** 2).sum()), sequences=[X])[0]
compute_norm_lines = th.function([X], results)

# how does this feature of diag work?

x = np.diag(np.arange(1,6, dtype=th.config.floatX), 1)
print compute_norm_lines(x)

# comparison with numpy

print np.sqrt((x ** 2).sum(1))

# computing the norms of columns of X

X = T.matrix()
results = th.scan(lambda x_i: T.sqrt((x_i ** 2).sum()), sequences = [X.T])[0]
compute_norm_cols = th.function([X], results)

x = np.diag(np.arange(1, 6, dtype=th.config.floatX), 1)
print compute_norm_cols(x)



