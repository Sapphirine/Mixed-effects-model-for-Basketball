import numpy as np
import pandas as pd
import patsy as ps
import scipy.sparse as sp
from patsy import dmatrices
import pickle
import time
from drop_columns_file import drop_columns
import csv
#########################################     DEFINE THE MODEL     ###################################################################################################################

# This function returns a list containing the fixed effect design matrix, it's column names and the matrices for the random effects with their column
# names and conditioning variables. 


#######################################################################################################################################
def specify_model(csv_file_name, ID_col_list, Y_flag):
  start_time = time.time()

  # set save_data = True if you want to save the design matrices and random effect matrices in a pickle file
  save_data = False;

  # Read the data from csv file
  data = pd.read_table(csv_file_name, sep = ',')
  print 'Data read from the file'

  Y_variable = 'repeatgr'
  # Select the columns of variables to make the data frame  
  vars_of_interest = [Y_variable, 'Minority', 'ses', 'schoolNR', 'ones'];

  vars_of_interest = vars_of_interest+ID_col_list

  # Change the integer/float variables which you want to treat as factor to string

  change_to_str = ['schoolNR']
  
  data[change_to_str] = data[change_to_str].astype(str)
  data = data[np.unique(vars_of_interest)]

  ####### Drop the rows with at least one NA

  data = data.dropna()
  data_ID_cols = data[ID_col_list]

  ###### Define the design matrix of fixed effect terms

  fixedmat = ps.dmatrix('''Minority + ses + ses * Minority''', data, return_type="dataframe")


  ########## List of column names of the design matrix

  fixedmat_column_names = fixedmat.design_info.column_names

  ######### Convert the design matrix from patsy data frame to scipy array 
  fixedmat = sp.csr_matrix(fixedmat.as_matrix())

  # Drop some columns
  list_of_columns_to_remove = []


  result = drop_columns(fixedmat, fixedmat_column_names, list_of_columns_to_remove)

  # fixedmat and it's columns names after dropping the columns mentioned above
  fixedmat = result[0]
  fixedmat_column_names = result[1]

  ### Check rank of fixedmat by looking the rank of temp matrix below
  temp = fixedmat.T*fixedmat;
  rank_fixedmat = np.linalg.matrix_rank(temp.toarray())
  if rank_fixedmat<fixedmat.shape[1]:
    print 'Fixedmat is low rank with rank = ' + repr(rank_fixedmat) + ', its full rank should be ' + repr(fixedmat.shape[1])

  ######### The prediction variable

  if Y_flag:
    Y = np.asarray(data[Y_variable].tolist())
    Y = np.reshape(Y, (len(Y),1))


  ################################################### Defining random effects ################################
  formula = '''0 + ones'''

  ##### First random effect ###########
  # design matrix of random effects
  X1 = ps.dmatrix(formula, data)
  # column names of random effects
  X1_column_names = X1.design_info.column_names
  X1 = np.asarray(X1.tolist())


  #### The conditioning variable of X1 random effect 
  condX1 = np.asarray(data['schoolNR'])

  ### List of random effect variables

  reData = [(X1, condX1, X1_column_names)]

  data={}; data['fixedmat'] = fixedmat; data['fixedmat_column_names'] = fixedmat_column_names; data['reData'] = reData; data['ID_cols'] = data_ID_cols
  if Y_flag:
    data['Y'] = Y; 
  if save_data:
    output  = open('data.pkl', 'wb'); 
    pickle.dump(data, output); 
    output.close()

  return(data)