from __future__ import division
import numpy as np
from scipy.sparse import *
from construct_Zp_file import construct_Zp
from findelementsfile import findelements

### Calculates covariance of random effects usign the sampled b of players with at least one observation for the random effect

def covblock(b,reData,obsreIDresult):
	result = [];
	i=0;
	for temp in reData:
		index = np.arange(len(np.unique(temp[1]))*temp[0].shape[1])
		bTemp = b[index];
		b = np.delete(b, index)
		bTempmat = np.reshape(bTemp.ravel(), (len(bTemp)/temp[0].shape[1], temp[0].shape[1]))
		pID = np.sort(np.unique(temp[1]))
		bTempmat = bTempmat[np.invert(findelements(pID, obsreIDresult[i])),:]
		bTempmat = bTempmat - np.mean(bTempmat,0)
		result.append(bTempmat.T.dot(bTempmat)/bTempmat.shape[0])
		i = i+1
	return (result)