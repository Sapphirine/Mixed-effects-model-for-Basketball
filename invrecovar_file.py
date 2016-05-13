from __future__ import division
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import block_diag
import scipy.sparse as sp

# Returns inverse covariance matrix of the b vector given the estimate of covariance matrix of all the random effects (estCov)
def invrecovar(reData, sampleCov):
	numre = len(reData)
	temp = []
	for i in range(numre):
		nop = len(np.unique(reData[i][1]))
		temp.append(sp.kron(sp.identity(nop),np.linalg.inv(sampleCov[i])));
	result = temp[0]
	if numre>1:
		for i in np.arange(1,numre):
			result = block_diag((result, temp[i]))
	return (result)