import numpy as np
# Returns unique elements of covariance matrices as a vector
def get_uniq_cov_elements(estCovs):
	result = []
	for estCov in estCovs:
		cov_matrix = np.reshape(estCov, (-1, estCov.shape[0]))
		cov_unique_elemets = cov_matrix[np.triu_indices(cov_matrix.shape[0])]
		result = np.append(result,cov_unique_elemets)
	return(result)
