import numpy as np
from scipy.sparse import *

### Helper function of construct_Z function
# X: array of dimension N x p
# player_id: 1D array of length N

# Z = coo matrix of size N x np (number of unique players)

def construct_Zp(X, player_id):
	if len(player_id.shape)>1:
		player_id = player_id.reshape(len(player_id),)
	temp = np.sort(np.unique(player_id))
	#print (temp)
	c = X.shape[1]*(np.searchsorted(temp,player_id))
	#print (c)
	Z = lil_matrix((X.shape[0], X.shape[1]*len(temp)))
	for iters in range(X.shape[1]):
		#print (c+iters)
		Z[(np.arange(X.shape[0]), c+iters)] = X[:,iters]
	return(Z) 