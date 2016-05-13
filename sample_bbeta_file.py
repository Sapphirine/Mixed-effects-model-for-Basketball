from __future__ import division
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scikits.sparse.cholmod import cholesky
import scipy.sparse as sp
import pickle
import time
import random as random
from sample_v_file import sample_v
import time

# Sample b's (refer to simulation paper)
def sample_bbeta(bbeta,hessian, Y, W, nore, inv_curr_covar, log_c1, num_bsamples, m_factor, num_v_samples, num_b_samples_per_try, parallel_loop_flag, num_parallel_loop):
	FLAG = True; # line 5
	# line 6 to 19
	while FLAG:
		FLAG = False;
		result = sample_v(sp.csr_matrix(hessian), m_factor, nore, bbeta, Y, W, inv_curr_covar, log_c1, num_v_samples, num_parallel_loop, parallel_loop_flag)
		v = result[0]
		if sum(v<0)>0:
			FLAG = True;
			print ('m_factor not good enough')
			m_factor = m_factor*1.02 # Choosing new proposal distribution	
	################## end of line 6 to 19

	v = np.append(sorted(v), float("inf")) # line 20
	
	##### Line 21 to 24
	q_v = np.arange(num_v_samples).T/num_v_samples
	temp = np.exp(-v);
	omega_vec = q_v*(temp[0:num_v_samples]-temp[1:(num_v_samples+1)]);
	
	##### end of line 21 to 24

	##### line 25 to 37 ###########
	for iters in range(num_bsamples): 
		j = ((np.random.multinomial(1, omega_vec/sum(omega_vec), size=1))[0]).nonzero()[0][0] 
		#j = random.randint(0,len(omega_vec)-1);
		eta = np.random.rand(1)[0]
		v_j_star = v[j]	-np.log(1-eta*(1-np.exp(v[j]-v[j+1])));
		v_j_star = np.ravel(v_j_star)
		v_j_star = np.squeeze(np.asarray(v_j_star))

		p = float("inf");
		pflag = True; #(False when we find at least one p < v*)
		while pflag:
			result = sample_v(sp.csr_matrix(hessian), m_factor, nore, bbeta, Y, W, inv_curr_covar, log_c1, num_b_samples_per_try, num_parallel_loop, parallel_loop_flag)
			p = result[0] # same as p (or v) samples
			bbeta_prop = result[1] # same as \theta_r samples
			# if we find at least one sample of p satisying p < v*, we choose the corresponding \theta_r (or b_prop) as the sample from posterior
			if sum(p<v_j_star)>0:
				pflag = False;
				bbeta_prop = bbeta_prop.toarray()
				bbeta_prop = bbeta_prop[:,(p<v_j_star).nonzero()[0][0]]
			else:
				print('bbeta sample rejected')

	return (bbeta_prop, m_factor)