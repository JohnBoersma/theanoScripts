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

# the second argument of diag puts it off the main diagonal

x = np.diag(np.arange(1,6, dtype=th.config.floatX), 1)
print compute_norm_lines(x)

# comparison with numpy

# sum on columns to get norm of rows

print np.sqrt((x ** 2).sum(1))

# computing the norms of columns of X

X = T.matrix()
results = th.scan(lambda x_i: T.sqrt((x_i ** 2).sum()), sequences = [X.T])[0]
compute_norm_cols = th.function([X], results)

x = np.diag(np.arange(1, 6, dtype=th.config.floatX), 1)
print compute_norm_cols(x)

# comparison with numpy

# sum on rows to get norm of columns

print np.sqrt((x ** 2).sum(0))

print '\nComputing a trace\n'

#th.config.floatX = 'float32'
floatX = th.config.floatX

X = T.matrix()

# cast changes the type of a tensor
# why needed?
# asked on stack overflow

results = th.scan(lambda i,j,t_f : X[i,j] + t_f,
                          sequences=[T.arange(X.shape[0]), T.arange(X.shape[1])],
                          outputs_info=np.asarray(0., dtype=floatX))[0]
result = results[-1]
compute_trace = th.function([X], result)

x = np.eye(5, dtype=floatX)
x[0] = np.arange(5, dtype=floatX)

print x
print compute_trace(x)

# compariison with numpy

print np.diagonal(x).sum()

print '\nx(t) = x(t-2).dot(U) + x(t-1).dot(V) + tanh(x(t-1).dot(W) + b)\n'

X = T.matrix()
W = T.matrix()
b_sym = T.vector()
U = T.matrix()
V = T.matrix()
n_sym = T.iscalar()

# the tutorail uses alternative dictionary syntax dict(initial=X, taps=[-2,-1])

results = th.scan(lambda x_tm2, x_tm1 : T.dot(x_tm2,U) + T.dot(x_tm1,V) + T.tanh(T.dot(x_tm1,W) + b_sym),
                          n_steps=n_sym, outputs_info=[{'initial':X, 'taps':[-2,-1]}])[0]
compute_seq2 = th.function([X,U,V,W,b_sym,n_sym], results)

x = np.zeros((2,2), dtype = th.config.floatX)
x[1,1] = 1
w = 0.5 * np.ones((2,2), dtype=th.config.floatX)
u = 0.5 * (np.ones((2,2), dtype=th.config.floatX) - np.eye(2, dtype=th.config.floatX))
v = 0.5 * np.ones((2,2), dtype=th.config.floatX)
n = 10
b = np.ones((2), dtype=th.config.floatX)

print compute_seq2(x,u,v,w,b,n)

# comparison with numpy

x_res = np.zeros((10,2))
x_res[0] = x[0].dot(u) + x[1].dot(v) + np.tanh(x[1].dot(w) + b)
x_res[1] = x[1].dot(u) + x_res[0].dot(v) + np.tanh(x_res[0].dot(w) + b)
x_res[2] = x_res[0].dot(u) + x_res[1].dot(v) + np.tanh(x_res[1].dot(w) + b)
for i in range(2,10):
    x_res[i] = (x_res[i-2].dot(u) + x_res[i-1].dot(v) + 
                     np.tanh(x_res[i-1].dot(w) + b ))
print x_res

print '\nJacobian of tanh example\n'

v = T.vector()
A = T.matrix()
y = T.tanh(T.dot(v,A))

# for some reason related to the scan internals
# we need to iterate over the indices of y rather
# than over the elements of y itself

results = th.scan(lambda i : T.grad(y[i], v), sequences=[T.arange(y.shape[0])])[0]
compute_jac_t = th.function([A,v], results)

x = np.eye(5, dtype=floatX)[0]

# second parameter of eye is the number
# of columns if different from the number of rows

w = np.eye(5,3, dtype=floatX)
w[2] = np.ones((3), dtype=floatX)
print compute_jac_t(w,x)

# compare with numpy

print ((1 - np.tanh(x.dot(w)) ** 2) * w).T

print '\nAccumulate number of loops during a scan'

# set initial value of shared variable

k = th.shared(0)
n_sym = T.iscalar()

# need to fix / understand from here

results, updates = th.scan(lambda : {k:(k+1)}, n_steps = n_sym)
accumulator = th.function([n_sym], [], updates = updates, allow_input_downcast=True )

print k.get_value()
accumulator(5)
print k.get_value()




