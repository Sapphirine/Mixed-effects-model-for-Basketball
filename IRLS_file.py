import numpy as np
import scipy.sparse as sp
import scipy as scipy
from scipy.sparse import block_diag
from invrecovar_file import invrecovar
import time
from scipy.sparse import block_diag
from scikits.sparse.cholmod import cholesky

# Estimates the mode of posterior of random effects and MLE estimate of fixed effects
# Uses newton's method

def IRLS(Y, W, nore, reData, estCov, curr_bbeta):
    N = W.shape[0]
    nofe = W.shape[1] - nore
    err = 1
    logistic_input = W*curr_bbeta;
    curr_pred = np.asarray(np.divide(np.exp(logistic_input),(1+np.exp(logistic_input))))
    inv_curr_covar = invrecovar(reData, estCov); # Construct the covariance matrix b given the estimate of covariance matrix of random effects
    hessian_from_logprior = block_diag((inv_curr_covar,np.zeros((nofe,nofe))))
    start_time = time.time()
    curr_b = curr_bbeta[np.arange(nore)] 
    while err>0.00001:
      old_b = curr_b;
      old_bbeta = curr_bbeta; 
      curr_gradient = -((W.T)*(Y-curr_pred) -sp.vstack(((inv_curr_covar*curr_b), np.zeros((nofe,1)))));
      hessain_from_loglik = (W.T)*(sp.csr_matrix(((curr_pred*(1-curr_pred)).ravel(), (np.arange(N), np.arange(N)))))*W;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
      curr_hessian = hessain_from_loglik+hessian_from_logprior;
      factor = cholesky(curr_hessian);
      delta_bbeta = factor(curr_gradient);
      delta_bbeta = np.reshape(delta_bbeta, (len(delta_bbeta),1));
      curr_bbeta = curr_bbeta - delta_bbeta;
      logistic_input = W*curr_bbeta;
      curr_pred = np.asarray(np.divide(np.exp(logistic_input),(1+np.exp(logistic_input))));
      curr_b = curr_bbeta[0:nore];
      curr_beta = curr_bbeta[nore:];
      err = np.sqrt(sum(np.square(curr_bbeta - old_bbeta))/len(old_bbeta));
      #print repr(err)
    pred_error = np.sqrt(sum(np.square(curr_pred - Y))/len(curr_pred)); # current prediction mean square error
    print 'Current training RMSE = '+repr(pred_error);
    hessian_from_loglik = (W.T)*(sp.csr_matrix(((curr_pred*(1-curr_pred)).ravel(), (np.arange(N), np.arange(N)))))*W;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    curr_hessian = hessian_from_logprior+hessian_from_loglik;
    curr_b = sp.csr_matrix(curr_b);
    log_c1 = sp.csr_matrix((W*curr_bbeta).T)*Y - sum(np.log(1+np.exp(W*curr_bbeta))) - 0.5*curr_b.T*(inv_curr_covar)*(curr_b); # refer to sampling paper
    # log_lik_part = - 0.5*inv_curr_covar.shape[1]*np.log(2*np.pi);

    # for i in range(len(reData)):
    #   p = len(np.unique(reData[i][1]));
    #   (sign, logdet) = np.linalg.slogdet(estCov[i]);
    #   log_lik_part = log_lik_part - p*logdet;

    # log_lik = log_c1 + log_lik_part;
    factor1 = cholesky(curr_hessian);
    factor2 = cholesky(inv_curr_covar);
    log_lik = -factor1.logdet()+factor2.logdet();

    return ([curr_bbeta, curr_pred, curr_hessian, log_c1, log_lik])