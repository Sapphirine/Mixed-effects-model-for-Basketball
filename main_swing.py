import numpy as np
import scipy.sparse as sp
import scipy as scipy
from scipy.sparse import block_diag
from covblock import covblock
import os
import pandas as pd
from sample_bbeta_file import sample_bbeta
from notobsreIDfile import notobsreID
import pickle
import time
from IRLS_file import IRLS
from save_bbeta_file import save_bbeta
from save_bbeta_samples_file import save_bbeta_samples
from get_uniq_cov_elements_file import *
from calc_mean_covs_file import *
from save_covs_file import *
from construct_Z_file import construct_Z
from specify_model_file_swing import specify_model
from gen_data import gen_artificial_data
from invrecovar_file import invrecovar

### Maximum absolute relative change in the elements of current sample mean of covariance matrices is used as convergence term
### 

start_time_first = time.time()
burn_in = 20
MAXITER = 100
convergence_threshold = 0
num_parallel_loop = 36
num_v_samples = 1000
num_b_proposals_per_try = 120;
parallel_loop_flag = True; # Set it False if you don't want to use parallel sampling
reload_parameters = False; # If you want the code to restart where it left off (useful when the code stopped running for some reason)
use_default_initial_estCov = True; # if False, initialize estCov below


### Choose gen_articial_data() to run the model on artificial data

data = specify_model('trData.csv', ['game_string','sv_pitch_id'], True) #gen_artificial_data()
fixedmat = data['fixedmat']; Y = data['Y'];reData = data['reData']; 
fixedmat_column_names = data['fixedmat_column_names']
Z = construct_Z(reData) # This construct the design matrix for the random effects given the random effect data
print 'Data read and Z Calculated'
notobsreIDresult = notobsreID(reData, num_parallel_loop, parallel_loop_flag);
print 'notobsreIDresult calculated'


N = fixedmat.shape[0]  # Number of data points
nore = Z.shape[1] # Number of random effects
W = sp.csr_matrix(sp.hstack((Z, fixedmat))); 
print repr(W.shape[1])

################################################### Learning the model ######################################################

##### Initializing variables

curr_bbeta = np.zeros((W.shape[1],1)) # Current estimate of [random_effect (b); fixed_effects (beta)] 

if use_default_initial_estCov:
  estCov = [] # Estimate of covariance of each random effect
  priorsd = 1; priorCorr = 0; # Initial estimate of standard deviation and correlation for each element of the covariance matrix
  # Construct the covariance matrix from the standard deviation and correlation values
  for i in range(len(reData)):
    p = reData[i][0].shape[1]
    temp = np.eye(p); temp[temp==0] = priorCorr; 
    sdRp = np.asarray([priorsd]*p); sdRp = np.reshape(sdRp, (len(sdRp),1)); estCov.append((sdRp.dot(sdRp.T))*temp);


##### Initializing the list to save different variables

sampled_bbetas = np.zeros((W.shape[1], MAXITER-burn_in)) # saves the mode of posterior of the effects from each iteration
estCovs_list = [] # store the covariances estimated in each iteration
conv_term_list = [] # saves the value of convergence term in each iteration
cov_ts = np.asarray([]) # saves distinct elements of all the covariance matrices in vectorized form (used to calculate the convergence term)
log_lik_list = []; # saves the log likelihood of each MAP posterior estimate

# Choosing proposal distribution is equivalent to choose m_factor (= 1/s, where s is the scaling factor for covariance)
# See last paragraph on page 8 in http://research.chicagobooth.edu/~/media/ADEE517350244BB2ACBE5EDA15E2E85A.pdf
m_factor = 1.02; # scaling term for inverse covariance of proposal distribution
conv_term = 10000; # current value of convergence term
iters = 0; # current iteration

if reload_parameters:
  data = pickle.load(open('inter_results_binary.p', "rb"))
  estCovs_list = data['cov_list']
  iters = len(estCovs_list)
  estCov = estCovs_list[-1]
  #print repr(estCov)
  sampled_bbetas = data['sampled_bbetas']
  curr_bbeta = sampled_bbetas[:,iters-burn_in-1] # Last estimate of [random_effect (b); fixed_effects (beta)] 
  curr_bbeta = np.reshape(curr_bbeta, (len(curr_bbeta),1));
  for estCov_iter in estCovs_list:
    uniq_cov_elements = get_uniq_cov_elements(estCov_iter)
    cov_ts = np.vstack((cov_ts,uniq_cov_elements)) if cov_ts.size else uniq_cov_elements


while ((iters>=MAXITER)+(conv_term<convergence_threshold)<1):
  
  # Find the mode and hessian of the posterior of b and the estimate of beta
  print 'iteration '+ repr(iters)
  start_time = time.time()
  result = IRLS(Y, W, nore, reData, estCov, curr_bbeta) # Estimates the mode of posterior of random effect and MLE estimate of beta
  #print 'Optimization took '+repr(time.time()-start_time) + ' secs'
  curr_bbeta = result[0] 
  curr_hessian = result[2]
  log_c1 = result[3] # log of c1 in line 4 of pseudo code
  log_lik_list.append(result[4]);
  
  start_time = time.time()
  num_bbetasamples = 1; # same as R in line 1 of pseudo code
  inv_curr_covar = invrecovar(reData, estCov);
  result_temp = sample_bbeta(sp.csr_matrix(curr_bbeta),curr_hessian, Y, W, nore, inv_curr_covar, log_c1, num_bbetasamples, m_factor, num_v_samples, num_b_proposals_per_try, parallel_loop_flag, num_parallel_loop)
  bbeta_sample = result_temp[0]
  m_factor = result_temp[1] # set m_factor equal to the m_factor that worked in last sampling
  b_sample = bbeta_sample[0:nore]
  #print 'Time taken to sample = '+repr(time.time()-start_time)

  # Estimate the covariance matrices of random effects given the sample of b
  estCov = covblock(b_sample, reData, notobsreIDresult) # Calculate covariance blocks given the sample for the random effects and return it as a list
  print(estCov)

  estCovs_list.append(estCov) # stores the list of covariance matrices estimated in each iteration
  
  # Save the intermediate results 
  f = open('inter_results_binary.p','wb');
  pickle.dump({'cov_list':estCovs_list, 'sampled_bbetas':sampled_bbetas},f); f.close();


  #### Calculate the convergence term in each iteration
  uniq_cov_elements = get_uniq_cov_elements(estCov)
  cov_ts = np.vstack((cov_ts,uniq_cov_elements)) if cov_ts.size else uniq_cov_elements

  iters = iters+1;

  if iters>burn_in:
    conv_term = max(np.absolute((np.mean(cov_ts[burn_in:iters,:],0)-np.mean(cov_ts[burn_in:(iters-1),:],0))/np.std(cov_ts[burn_in:iters,:],0))) 
    conv_term_list.append(conv_term)
    sampled_bbetas[:,iters-burn_in-1] = bbeta_sample;
    #print 'Max absolute relative change in mean of covariance elements is: '+repr(conv_term)
#######################################################

# print (burn_in+np.argmax(log_lik_list[burn_in:]))
# curr_bbeta = sampled_bbetas[:,burn_in+np.argmax(log_lik_list[burn_in:])];
# curr_bbeta = np.reshape(curr_bbeta, (len(curr_bbeta),1));
# final_ecov = estCovs_list[burn_in+np.argmax(log_lik_list[burn_in:])];
# print (log_lik_list);

bprediction = 0;
for i in range(sampled_bbetas.shape[1]):
  logistic_input = W*sampled_bbetas[:,i]
  bprediction = bprediction + np.asarray(np.divide(np.exp(logistic_input),(1+np.exp(logistic_input))))
bprediction = bprediction/sampled_bbetas.shape[1] # bayesian prediction

final_ecov = calc_mean_covs(estCovs_list, burn_in) # [np.reshape(np.asarray([0.258]),(1,1))];# mean of the covariances sampled, #[np.reshape(np.asarray([0.258]),(1,1))];
result = IRLS(Y, W, nore, reData, final_ecov, curr_bbeta) # b and beta estimates using final covariance estiamte
curr_bbeta = result[0]

# Save the predictons in a csv file
logistic_input = W*curr_bbeta
pprediction = np.asarray(np.divide(np.exp(logistic_input),(1+np.exp(logistic_input)))) #pointwise estimate prediction

print 'Final point training RMSE = '+repr(np.sqrt(sum(np.square(bprediction.ravel() - Y.ravel()))/len(bprediction)))
print 'Final bayesian training RMSE = '+repr(np.sqrt(sum(np.square(pprediction.ravel() - Y.ravel()))/len(pprediction)))



pprediction = pd.DataFrame(pprediction)
ID_cols = data['ID_cols']
ID_cols['pprediction'] = pd.DataFrame(pprediction)
ID_cols.to_csv('training_pprediction.csv', index=False)

bprediction = pd.DataFrame(bprediction)
ID_cols = data['ID_cols']
ID_cols['bprediction'] = pd.DataFrame(bprediction)
ID_cols.to_csv('training_bprediction.csv', index=False)

save_bbeta(curr_bbeta, reData, W, nore, fixedmat_column_names)
save_bbeta_samples(sampled_bbetas, reData, W, nore, fixedmat_column_names);
save_covs(final_ecov, reData)
f = open('results_binary.pkl','wb');
pickle.dump({'cov_list':estCovs_list, 'estCov':final_ecov, 'sampled_bbetas':sampled_bbetas, 'conv_term':conv_term_list, 'bbeta':curr_bbeta},f); f.close();
print 'Total number of iterations taken is '+repr(iters)
print 'Time taken to learn b = '+repr(time.time()-start_time_first)