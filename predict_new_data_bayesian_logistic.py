import numpy as np
import pandas as pd
import patsy as ps
import scipy.sparse as sp
from patsy import dmatrices
import pickle
import time
from drop_columns_file import drop_columns
from specify_model_glmer_example import specify_model

# Predicts on the data set in file 'data_file' and saves the prediction iin 'test_predictions.csv'
def predict_new_data_bayesian(data_file, ID_column_list):

  data = specify_model(data_file, ID_column_list, False)
  fixedmat = data['fixedmat']; reData = data['reData']; fixedmat_column_names = data['fixedmat_column_names']
  bbeta_dict_samples = pickle.load(open('bbeta_dict_samples.pkl','rb'))
  prediction = 0;
  for i in range(len(bbeta_dict_samples)):
    bbeta_dict = bbeta_dict_samples[i];
    beta_dict = bbeta_dict['beta']

    beta = np.asarray([])
    for col_name in fixedmat_column_names:
      beta = np.append(beta, beta_dict[col_name])

    logistic_input = fixedmat*beta
    b_dict_list = bbeta_dict['b']

    for i in range(fixedmat.shape[0]):
      for (reData_group, reData_iter) in enumerate(reData):
        b_dict = b_dict_list[str(reData_group)]
        if reData_iter[1][i] in b_dict.keys():
          logistic_input[i] = logistic_input[i]+b_dict[reData_iter[1][i]].dot(reData_iter[0][i,:])
        else:
          print 'Player ' + repr(reData_iter[1][i]) + ' Coefficient_name = ' + repr(col_name) + ' in group '+ repr(reData_group) + ' has no value observed'

    prediction = prediction+np.divide(np.exp(logistic_input), (1+np.exp(logistic_input)))

  prediction = prediction/len(bbeta_dict_samples)

  ID_cols = data['ID_cols']
  ID_cols['prediction'] = pd.DataFrame(prediction)
  ID_cols.to_csv('test_prediction_bayesian.csv', index=False)

