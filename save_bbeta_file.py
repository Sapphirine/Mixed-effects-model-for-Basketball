import numpy as np
import csv
import pickle as pickle
	
## This function will work in python2
# Saves the random effects and fixed effects in a csv file and also in a pickle file to be used later for prediction
# X_info is the fixedmat_column_naes

def save_bbeta(bbeta, reData, W, nore, X_info):
	b = bbeta[0:nore]
	beta = bbeta[nore:(W.shape[1])]
	result = [];
	bbeta_dict = {};
	beta_dict = {};
	b_dict_list = {};
	with open('results_re.csv', 'wb') as csvfile:
		rewriter = csv.writer(csvfile, delimiter=',',quotechar='''\"''', quoting=csv.QUOTE_MINIMAL)
		rewriter.writerow(['Group', 'Player_id', 'Coefficient_name', 'value'])
		for (iter_reData, elements_reData) in enumerate(reData):
			b_dict = {};
			index = np.arange(len(np.unique(elements_reData[1]))*elements_reData[0].shape[1])
			bTemp = b[index];
			b = np.delete(b, index)
			bTempmat = np.reshape(bTemp.ravel(), (len(bTemp)/elements_reData[0].shape[1], elements_reData[0].shape[1]))
			pID = np.sort(np.unique(elements_reData[1]))
			reData_info = elements_reData[2]
			for (pid_iter, pid) in enumerate(pID):
				b_for_pid = bTempmat[pid_iter,:]
				b_dict[pid] = b_for_pid
				for (iter_b_for_pid, elements) in enumerate(b_for_pid):
					rewriter.writerow([iter_reData, pid, reData_info[iter_b_for_pid], elements])
			b_dict_list[str(iter_reData)] = b_dict

	bbeta_dict['b'] = b_dict_list

	with open('results_beta.csv', 'wb') as csvfile:
		rewriter = csv.writer(csvfile, delimiter=',',quotechar='''\"''', quoting=csv.QUOTE_MINIMAL)
		rewriter.writerow(['Coefficient_name', 'value'])
		for (beta_iter, beta_value) in enumerate(beta):
			beta_dict[X_info[beta_iter]] = beta_value[0]
			rewriter.writerow([X_info[beta_iter], beta_value[0]])

	bbeta_dict['beta'] = beta_dict
	output  = open('bbeta_dict.pkl', 'wb'); 
	pickle.dump(bbeta_dict, output); 
	output.close()

	# logistic_input = W*curr_bbeta;
	# curr_pred = np.asarray(np.divide(np.exp(logistic_input),(1+np.exp(logistic_input))))
	# with open('results_prediction.csv', 'wb') as csvfile:
	# 	rewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	# 	rewriter.writerow(['game_string','sv_pitch_id','Prediction'])
	# 	for (pred_iter, pred_value) in enumerate(curr_pred):
	# 		rewriter.writerow([game_string[pred_iter],sv_pitch_id[pred_iter], pred_value[0]])




