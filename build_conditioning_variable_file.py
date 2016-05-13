import patsy as ps
import numpy as np
# Construct IDs for conditioning varaibles given the formula

def build_conditioning_var(formula, data):
	result = []
	start = 0; end = 0; 
	while end<len(data):
		print(end)
		start = end; end = end+10000
		if end>len(data):
			end = len(data)
		temp = ps.dmatrix(formula, data[start:end])
		rows, cols = temp.nonzero()
		result.extend([temp.design_info.column_names[i] for i in cols])


	return(np.asarray(result))

