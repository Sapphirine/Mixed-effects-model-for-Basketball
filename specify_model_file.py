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
  data = pd.read_table(csv_file_name,sep=',')
  print 'Data read from the file'

  Y_variable = "FGM"
  # Select the columns of variables to make the data frame  
  vars_of_interest = [Y_variable,"MINUTES_REMAINING","CLOSEST_DEFENDER_PLAYER_ID", "PTS_TYPE", "ACTION_TYPE", "LOCATION", "SHOT_DISTANCE", "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME", "CLOSE_DEF_DIST", "center3", "paint", "leftCorner3", "rightCorner3", "midrange", "restricted", "rowNumber", "PLAYER_ID"];

  # Change the integer/float variables which you want to treat as factor to string

  change_to_str = ["PLAYER_ID","CLOSEST_DEFENDER_PLAYER_ID"]
  data[change_to_str] = data[change_to_str].astype(str)
  data = data[np.unique(vars_of_interest)]

  ####### Drop the rows with at least one NA

  data = data.dropna()
  data_ID_cols = data[ID_col_list]


  ###### Define the design matrix of fixed effect terms

  fixedmat = ps.dmatrix('''center3+paint+leftCorner3+rightCorner3+midrange+restricted+MINUTES_REMAINING + PTS_TYPE + 
    SHOT_DISTANCE + SHOT_CLOCK + LOCATION + ACTION_TYPE + DRIBBLES + TOUCH_TIME + CLOSE_DEF_DIST + DRIBBLES*TOUCH_TIME + SHOT_DISTANCE*PTS_TYPE ''', data)


  ########## List of column names of the design matrix

  fixedmat_column_names = fixedmat.design_info.column_names

  ######### Convert the design matrix from patsy data frame to scipy array 
  fixedmat = fixedmat.tolist()
  fixedmat = sp.csr_matrix(np.asarray(fixedmat))

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
  formula = '''0+center3+paint+leftCorner3+rightCorner3+midrange+restricted'''

  ##### First random effect ###########
  X1 = ps.dmatrix(formula, data)
  X1_column_names = X1.design_info.column_names
  X1 = np.asarray(X1.tolist())

  #### THe conditioning variable of X1 random effect (a column of string/factor representing the player ID or whatever the conditioning variable is)
  ### Define condX1 as below if it's one of the columns of the data matrix. 
  #condX1 = np.asarray(data['BatterMlbId'].tolist()) 
  condX1 = np.asarray(data['PLAYER_ID'].tolist())
  ####### End of definintion of first random effect ###########


  #### Repeat like above to define more random effects ####

  formula = '''0+center3+paint+leftCorner3+rightCorner3+midrange+restricted'''
  X2 = ps.dmatrix(formula, data)
  X2_column_names = X2.design_info.column_names
  X2 = np.asarray(X2.tolist())
  condX2 = np.asarray(data["CLOSEST_DEFENDER_PLAYER_ID"].tolist())
  #condX2 = np.asarray(data['BatterMlbId'].tolist())

  ### List of random effect variables

  reData = [(X1, condX1, X1_column_names), (X2, condX2, X2_column_names)]

  data={}; data['fixedmat'] = fixedmat; data['fixedmat_column_names'] = fixedmat_column_names; data['reData'] = reData; data['ID_cols'] = data_ID_cols
  if Y_flag:
    data['Y'] = Y; 
  if save_data:
    output  = open('data.pkl', 'wb'); 
    pickle.dump(data, output); 
    output.close()

  return(data)