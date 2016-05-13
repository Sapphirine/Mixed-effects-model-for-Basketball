import numpy as np
# REmoes the columns in fixedmat that are in drop_columns_list and return new list of columns and fixedmat
def drop_columns(fixedmat, fixedmat_column_names, drop_column_list):
	for column_name in drop_column_list:
		all_cols = np.arange(len(fixedmat_column_names))
		cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, fixedmat_column_names.index(column_name))))[0]
		fixedmat = fixedmat[:, cols_to_keep]
		fixedmat_column_names.remove(column_name)
		print repr(column_name) + ' removed'
	return ([fixedmat, fixedmat_column_names])
			
