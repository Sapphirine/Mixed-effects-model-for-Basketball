import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy as scipy
from scipy.sparse import block_diag
from covblock import covblock
import os
import pandas as pd
from sample_b_file import sample_b
from notobsreIDfile import notobsreID
import pickle
import time
from IRLS_file import IRLS
from save_bbeta_file import save_bbeta
from get_uniq_cov_elements_file import *
from calc_mean_covs_file import *
from save_covs_file import *
from construct_Z_file import construct_Z
from specify_model_file_swingdata import specify_model
from gen_data import gen_artificial_data
from invrecovar_file import invrecovar

####   This function can be used to get results from the intermediate samples saved in 'inter_results_binary.p'

### Maximum absolute relative change in the elements of current sample mean of covariance matrices is used as convergence term

start_time_first = time.time()
burn_in = 1
MAXITER = 5
convergence_threshold = 0.05
num_parallel_loop = 12
num_v_samples = 1000
num_b_proposals_per_try = 120;
parallel_loop_flag = True; # Set it False if you don't want to use parallel sampling
reload_parameters = False; # If you want the code to restart where it left off (useful when the code stopped running for some reason)
use_default_initial_estCov = True; # if False, initialize estCov below


data = specify_model('swingData2012-2.csv', ['pitchFXid'], True) 
fixedmat = data['fixedmat']; Y = data['Y'];reData = data['reData']; 
fixedmat_column_names = data['fixedmat_column_names']
Z = construct_Z(reData) # This construct the design matrix for the random effects given the random effect data
print 'Data read and Z Calculated'
notobsreIDresult = notobsreID(reData, num_parallel_loop, parallel_loop_flag);
print 'notobsreIDresult calculated'


N = fixedmat.shape[0]  # Number of data points
nore = Z.shape[1] # Number of random effects
W = sp.csr_matrix(sp.hstack((Z, fixedmat))); 

result_data = pickle.load(open('inter_results_binary.p', "rb"))
estCovs_list = result_data['cov_list']
iters = len(estCovs_list)
sampled_bbetas = result_data['sampled_bbetas']
curr_bbeta = sampled_bbetas[:,iters-1] # Last estimate of [random_effect (b); fixed_effects (beta)] 
curr_bbeta = np.reshape(curr_bbeta, (len(curr_bbeta),1));

final_ecov = calc_mean_covs(estCovs_list, burn_in) # mean of the covariances sampled
result = IRLS(Y, W, nore, reData, final_ecov, curr_bbeta) # b and beta estimates using final covariance estiamte
curr_bbeta = result[0]

# Save the predictons in a csv file
logistic_input = W*curr_bbeta
prediction = np.asarray(np.divide(np.exp(logistic_input),(1+np.exp(logistic_input))))

print 'Final training RMSE = '+repr(np.sqrt(sum(np.square(prediction.ravel() - Y.ravel()))/len(prediction)))

prediction = pd.DataFrame(prediction)
ID_cols = data['ID_cols']
ID_cols['prediction'] = pd.DataFrame(prediction)
ID_cols.to_csv('training_prediction.csv', index=False)

save_bbeta(curr_bbeta, reData, W, nore, fixedmat_column_names)
save_covs(final_ecov, reData)
f = open('results_binary.pkl','wb');
pickle.dump({'cov_list':estCovs_list, 'estCov':final_ecov, 'sampled_bbetas':sampled_bbetas, 'bbeta':curr_bbeta},f); f.close();