import numpy as np
# calculates v_m (please refer to sampling paper)
### Note that we don't need log c_2 since it cancels out with log of denominator of g(\theta_m)
def calc_vm(W, Y, bbeta_prop, bbeta, inv_curr_covar, hessian,log_c1, m_factor, nore):
	v_m = np.zeros((bbeta_prop.shape[1],1))
	for i in range(bbeta_prop.shape[1]):
		Wbbeta_propi = W*bbeta_prop[:,i]
		b_propi = bbeta_prop[0:nore,i]
		v_m[i] = -((Wbbeta_propi).T*Y - np.sum(np.log(1+np.exp(Wbbeta_propi.toarray())))-0.5*b_propi.T*inv_curr_covar*b_propi+0.5*(bbeta_prop[:,i]-bbeta).T*(hessian/m_factor)*(bbeta_prop[:,i]-bbeta)-log_c1)
	return (v_m)