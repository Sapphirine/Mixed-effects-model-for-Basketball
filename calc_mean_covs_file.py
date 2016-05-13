import numpy as np
# Returns the mean of the sampled covariances after burn_in (start_index)
def calc_mean_covs(estCovs_list, start_index):
	result = [];
	for (iter_estCovs_list, estCovs) in enumerate(estCovs_list):

		if iter_estCovs_list>=start_index:
			if not result:
				result = estCovs
			else:
				for (iter_estCovs, estCov) in enumerate(estCovs):
					result[iter_estCovs] = result[iter_estCovs]+estCov
	for (iter_result, meanCov) in enumerate(result):
		result[iter_result] = meanCov/(len(estCovs_list)-start_index)

	return(result)