import numpy as np
# Return a list of length number of radnom effects group, each element of the list containts the player iDs which don't have any observation for that random effect group 
def notobsreID_help(data, pID_vec):
	temp = []
	for pID in pID_vec:
		X = data[0][data[1]==pID,]
		FLAG = True;
		for i in range(X.shape[1]):
			if sum(X[:,i])==0:
				FLAG = False
		if not FLAG:
			temp.append(pID)

	return (temp)
