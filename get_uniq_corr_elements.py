import numpy as np
def get_uniq_corr_elements(estCovs):
	result = []
	for estCov in estCovs:
		cov_matrix = np.reshape(estCov, (-1, estCov.shape[0]))
		sdevs = np.sqrt(np.diagonal(cov_matrix))
		sdevs = np.reshape(sdevs,(len(sdevs),1))
		corr_matrix = cov_matrix/(sdevs*sdevs.T)
		corr_unique_elements = corr_matrix[np.triu_indices(corr_matrix.shape[0],1)]
		result = np.append(result,corr_unique_elements)
	return(result)
