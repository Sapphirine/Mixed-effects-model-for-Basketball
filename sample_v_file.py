#from __future__ import division
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import scipy.linalg as sl	
from scikits.sparse.cholmod import cholesky
import time
from joblib import Parallel, delayed
import warnings
from calc_vm_file import calc_vm

# Sample v's (refer to simulation paper) line 10 to 13, except we draw all the samples at once and then check the validity condition in line 14 for all the samples
def sample_v(hessian, m_factor, nore, bbeta, Y, W, inv_curr_covar, log_c1, num_v_samples, num_parallel_loop, parallel_loop_flag):
	##### b_opt is same as \theta*
	
	# First we sample b's using cholesky decomposition of hessian
	#### start of line 11
	np.random.seed()
	start_time = time.time()
	warnings.filterwarnings("ignore")
	factor = cholesky(hessian)
	D = factor.D(); D = np.reshape(D, (len(D),1));
	bbeta_propP = np.sqrt(m_factor)*factor.solve_Lt(np.random.normal(size=(W.shape[1],num_v_samples))/np.sqrt(D))
	bbeta_prop = np.repeat(bbeta.toarray(), num_v_samples, axis=1)+ factor.apply_Pt(bbeta_propP)
	bbeta_prop = sp.csr_matrix(bbeta_prop, dtype=np.float64)
	##### end of line 11
	
	##### Calculate v's in parallel given the \theta's (b_prop)
	###### start of line 12

	if parallel_loop_flag:
		range_list = []
		for i in range(num_parallel_loop):
			range_list.append(np.arange(int(i*num_v_samples/num_parallel_loop), int((i+1)*num_v_samples/num_parallel_loop)))
		bbeta_prop_list = []
		for i in range(num_parallel_loop):
			bbeta_prop_list.append(bbeta_prop[:,range_list[i]])
		result = (Parallel(n_jobs=num_parallel_loop)(delayed(calc_vm)(W, Y, bbeta_prop_list[i], bbeta.toarray(), inv_curr_covar, hessian,log_c1, m_factor, nore) for i in range(num_parallel_loop)))
		v_m = ();
		for temp in result:
			v_m = v_m+(temp,)

		v_m = np.vstack(v_m)
		v_m = v_m.ravel()
		v_m = np.squeeze(np.asarray(v_m))
	else:
		result = calc_vm(W, Y, bbeta_prop, bbeta.toarray(), inv_curr_covar, hessian,log_c1, m_factor, nore)
		v_m = np.asarray(result); 
		v_m = np.reshape(v_m, (len(v_m),))
	
	############ End of line 12
	return (v_m, bbeta_prop)