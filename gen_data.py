import numpy as np
import scipy.sparse as sp
import os
import pickle
import time
from construct_Z_file import construct_Z


def gen_artificial_data():
	# Running the code on artificialy generated data
	ARTF_DATA = 1
	ARTF_GENERATE = 1
	BINARY_DATA = 1
	FILENAME_BDATA = 'bdata.p'
	FILENAME_LDATA = 'ldata.p'
	FILENAME_RESULT = 'result.pkl'
	np.random.seed(3)

	if ARTF_DATA:
		if ARTF_GENERATE:
			# problem data
			N = 200000; var = 0.1; nop, nob = 20,20;#1003, 1118;
			norb, nolb, norp, nolp = int(round(nob*0.65)), nob-int(round(nob*0.65)), int(round(nop*0.72)), nop-int(round(nop*0.72));
			p = 1

			if BINARY_DATA==1:
				sdRp = np.array([[0.3150],[0.3128]]); covRp = np.dot(sdRp,sdRp.transpose())*np.array([[1,0.78],[0.78,1]])#0.78
				sdLp = np.array([[0.3335],[0.2886]]); covLp = np.dot(sdLp,sdLp.transpose())*np.array([[1,0.57],[0.57,1]])#0.57
				sdLb = np.array([[0.3837],[0.3972]]); covLb = np.dot(sdLb,sdLb.transpose())*np.array([[1,0.94],[0.94,1]])#0.94
				sdRb = np.array([[0.4004],[0.3707]]); covRb = np.dot(sdRb,sdRb.transpose())*np.array([[1,0.93],[0.93,1]])#0.93
				#rp, rb, lp, lb
			else:
				sdRp = np.array([[0.015],[0.02]]); covRp = np.dot(sdRp,sdRp.transpose())*np.array([[1,0.8],[0.8,1]])
				sdLp = np.array([[0.02],[0.015]]); covLp = np.dot(sdLp,sdLp.transpose())*np.array([[1,0.75],[0.75,1]])
				sdLb = np.array([[0.02],[0.015]]); covLb = np.dot(sdLb,sdLb.transpose())*np.array([[1,0.85],[0.85,1]])
				sdRb = np.array([[0.015],[0.02]]); covRb = np.dot(sdRb,sdRb.transpose())*np.array([[1,0.9],[0.9,1]])

			bRp = np.random.multivariate_normal(np.zeros((2,)),covRp,norp)
			bRp = bRp.reshape((2*norp,1))
			bLp = np.random.multivariate_normal(np.zeros((2,)),covLp,nolp)
			bLp = bLp.reshape((2*nolp,1))
			bp = np.vstack((bRp, bLp))
			bRb = np.random.multivariate_normal(np.zeros((2,)),covRb,norb)
			bRb = bRb.reshape((2*norb,1))
			bLb = np.random.multivariate_normal(np.zeros((2,)),covLb,nolb)
			bLb = bLb.reshape((2*nolb,1))
			bb = np.vstack((bRb, bLb))
			b_true = np.vstack((bp, bp, bb, bb))

			bId = np.zeros((N,)); pId = np.zeros((N,))

			X1 = np.zeros((N,2)); X2 = np.zeros((N,2)); X3 = np.zeros((N,2)); X4 = np.zeros((N,2)); #order is nrp, nlp, nrb, nlb
			for i in range(N):
				batter = (np.random.rand(nob,1)).argmax(axis=0); bId[i] = str(int(batter));
				pitcher = (np.random.rand(nop,1)).argmax(axis=0); pId[i] = str(int(pitcher)); #print((pitcher, batter));
				lb = batter>=norb; rb = not lb;
				lp = pitcher>=norp; rp = not lp;
				if rp:
					if rb:
						X1[i,0] = 1
					else:
						X1[i,1] = 1
				else:
					if rb:
						X2[i,0] = 1
					else:
						X2[i,1] = 1
				if rb:
					if rp:
						X3[i,0] = 1
					else:
						X3[i,1] = 1
				else:
					if rp:
						X4[i,0] = 1
					else:
						X4[i,1] = 1


			#X1 = np.random.rand(N,2); X2 = np.random.rand(N,2); X3 = np.random.rand(N,2); X4 = np.random.rand(N,2);
			col_names = ['col1', 'col2']
			fixedmat_col_names = ['X1', 'X2']
			reData= [(X1, pId, col_names), (X2, pId, col_names), (X3, bId, col_names),(X4, bId, col_names)]
			X = np.random.normal(0,1,(N,p+1));
			X[:,p] = 1; 
			X = sp.csr_matrix(X)

			beta_true = np.random.rand(p,1);
			beta_true = np.vstack((beta_true,-1.2659));
			Z = construct_Z(reData)
			#print (Z.shape); print (b_true.shape)

			if BINARY_DATA==1:
				logistic_input = Z*b_true+X*beta_true;
				Y = np.random.rand(N,1)<np.divide(np.exp(logistic_input), (1+np.exp(logistic_input)))
				print 'Data generated!'
				#f = open(FILENAME_BDATA,'wb');
			else:
				Y = Z*b_true+X*beta_true+var*np.random.randn(N,1);
				#f = open(FILENAME_LDATA,'wb');
			data = {'fixedmat': X, 'Y': Y,'reData': reData,'fixedmat_column_names':fixedmat_col_names, 'nolp':nolp, 'nolb':nolb, 'norp':norp, 'norb':norb, 'bRp': bRp, 'bLp':bLp, 'bRb':bRb, 'bLb':bLb, 'beta_true':beta_true, 'var':var, 'covLp':covLp, 'covLb':covLb, 'covRp':covRp, 'covRb':covRb}

	return (data)