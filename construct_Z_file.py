import numpy as np
from scipy.sparse import *
from construct_Zp_file import construct_Zp
# constructs the design matrix for random effects given their values and the conditioning variables 

def construct_Z(data):
	Z = construct_Zp(data[0][0],data[0][1])
	for temp in data[1:len(data)]:
		arfd = construct_Zp(temp[0],temp[1])
		Z = hstack((Z,arfd))
	return(Z)
