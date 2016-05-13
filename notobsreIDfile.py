from notobsreID_help_file import *
import numpy as np
from joblib import Parallel, delayed

# notusefulID correspond to the player ID (or the conditioning variable in general) which have no observation in at least one of the columns (or random effet). 

def notobsreID(reData, num_parallel_loop, parallel_loop_flag):
	notusefulID = []
	for data in reData:
		unique_pID = np.unique(data[1])
		num_unique_pID = len(unique_pID)
		if parallel_loop_flag:
			range_list = []
			for i in range(num_parallel_loop):
				range_list.append(np.arange(int(i*num_unique_pID/num_parallel_loop), int((i+1)*num_unique_pID/num_parallel_loop)))
			unique_pID_list = []
			for i in range(num_parallel_loop):
				unique_pID_list.append(unique_pID[range_list[i]])
			result = (Parallel(n_jobs=num_parallel_loop)(delayed(notobsreID_help)(data, unique_pID_list[i]) for i in range(num_parallel_loop)))
			notusefulIDpart = [];
			for temp in result:
				notusefulIDpart.extend(temp)
		else:
			notusefulIDpart = notobsreID_help(data, unique_pID)
		
		notusefulID.append(np.asarray(notusefulIDpart))
	return(notusefulID)