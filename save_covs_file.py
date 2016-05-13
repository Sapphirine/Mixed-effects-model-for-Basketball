import csv
import numpy as np
# Save the covariance of random effects in csv file
def save_covs(estCovs, reData):
	with open('results_cov.csv', 'wb') as csvfile:
		rewriter = csv.writer(csvfile, delimiter=',',quotechar='''\"''', quoting=csv.QUOTE_MINIMAL)
		for (iter_cov, ecov) in enumerate(estCovs):
			rewriter.writerow(reData[iter_cov][2])
			for rows in range(ecov.shape[0]):
				rewriter.writerow(ecov[rows,:].tolist())
