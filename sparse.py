''' examples of sparse matrices'''

import numpy as np
import scipy.sparse as sp
import theano
from theano import sparse

# pylint: disable = bad-whitespace, invalid-name, no-member, bad-continuation, assignment-from-no-return

# if shape[0] > shape[1], use csr. Otherwise, use csc
# but, not all ops are available for both yet
# so use the one that has what you need

# to and fro

x = sparse.csc_matrix(name='x', dtype='float32')
y = sparse.dense_from_sparse(x)
z = sparse.csc_from_dense(y)

# resconstruct a csc from a csr

x = sparse.csc_matrix(name='x', dtype='int64')
data, indices, indptr, shape = sparse.csm_properties(x)
y = sparse.CSR(data, indices, indptr, shape)
f = theano.function([x], y)
a = sp.csc_matrix(np.asarray([[0,1,1], [0,0,0], [1,0,0]]))
print a.toarray()
print f(a).toarray()

# "structured" operations
# act only on (originally) nonzero elements

x = sparse.csc_matrix(name='x', dtype='float32')
y = sparse.structured_add(x,2)
f = theano.function([x], y)
a = sp.csc_matrix(np.asarray([[0,0,-1], [0,-2,1], [3,0,0]], dtype='float32'))
print a.toarray()
print f(a).toarray()
print f(f(a)).toarray()
