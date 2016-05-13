import numpy as np
# Returns a boolean vector of which elements of arr1 are in arr2
def findelements(arr1,arr2):
	result = []
	if arr2.size:
		for i in arr1:
			result.append(sum(arr2==i)>0)
		return(np.array(result))
	else:
		return([False] * len(arr1))