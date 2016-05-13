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
from specify_model_file import specify_model
from gen_data import gen_artificial_data
from invrecovar_file import invrecovar


data = pickle.load( open( "results_binary.pkl", "rb" ) )
cov_list = data['cov_list']
cov_ts = np.asarray([])

for estCov in cov_list:
	uniq_cov_elements = get_uniq_cov_elements(estCov)
	cov_ts = np.vstack((cov_ts,uniq_cov_elements)) if cov_ts.size else uniq_cov_elements

np.savetxt("cov_ts.csv", cov_ts, delimiter=",")
