import numpy as np
import scipy.sparse as sp
import os
import csv
import pickle
import time
from construct_Z_file import construct_Z
from construct_Zp_file import construct_Zp

#def gen_artificial_data():
	# Running the code on artificialy generated data
BINARY_DATA = 1
np.random.seed(4)
N = 10000;
noise_sigma = 0.2;
num_fixed_effects = 5;
num_random_effects = 4;
dim_random_effects = [1,2,3,4];
cov_random_effects = [];
corr_random_effects = [];
corr_random_effects.append(1);
corr_random_effects.append(np.array([[1, -0.9], [-0.9, 1]]));
corr_random_effects.append(np.array([[1, 0.1, 0.9], [0.1, 1, -0.9], [0.9, -0.9, 1]]));
corr_random_effects.append(np.array([[1, 0.5, -0.5, -0.1], [0.5, 1, 0.9, 0.05], [-0.5, 0.9, 1, 0], [-0.1, 0.05, 0, 1]]))

for i in range(num_random_effects):
	sdevs = np.random.uniform(0.3, 0.7, size=(dim_random_effects[i],1));
	cov = np.dot(sdevs,sdevs.transpose())*corr_random_effects[i];
	cov_random_effects.append(cov);

nop = 1000;
fixed_effects = np.random.uniform(0,1,size=(num_fixed_effects,1))
fixed_effects_X = sp.csr_matrix(np.random.normal(0,2,size=(N,num_fixed_effects)))
parameter_data = np.array([]);
random_effects = [];
for i in range(num_random_effects):
	print(i); print(np.linalg.matrix_rank(cov_random_effects[i]))
	temp = np.random.multivariate_normal(np.zeros((dim_random_effects[i],)),cov_random_effects[i],nop);
	random_effects.append(temp.reshape((dim_random_effects[i]*nop,1)));
	# if i==0:
	# 	parameter_data = temp.reshape((dim_random_effects[i]*nop,1))
	# else:
	# 	parameter_data = np.hstack((parameter_data, temp.reshape((dim_random_effects[i]*nop,1))))

pID = np.random.randint(nop, size=(N, 1)).astype('str')
parameter_data = np.hstack((parameter_data, pID));

linear_input = fixed_effects_X*fixed_effects;
full_X = fixed_effects_X;
for i in range(num_random_effects):
	random_effects_X = sp.csr_matrix(np.random.normal(0,2,size=(N,dim_random_effects[i])))
	full_X = sp.hstack([full_X, random_effects_X]);
	Zp = construct_Zp(random_effects_X.toarray(),pID)
	linear_input = linear_input+Zp*random_effects[i];
rescaled_full_X = full_X/np.std(linear_input)
rescaled_linear_input = linear_input/np.std(linear_input)


if BINARY_DATA==1:
	probabilities = np.divide(np.exp(linear_input), (1+np.exp(linear_input)))
	Y = np.random.rand(N,1)<probabilities
	print (np.mean(probabilities*(1-probabilities)))
else:
	Y = rescaled_linear_input+noise_sigma*np.random.normal(0,1,size=(N,1))

observed_data = np.hstack((rescaled_full_X.toarray(), pID));
observed_data = np.hstack((observed_data, Y));

with open('data.csv', 'w', newline='') as csvfile:
	rewriter = csv.writer(csvfile, delimiter=',',quotechar='''\"''', quoting=csv.QUOTE_MINIMAL)
	rewriter.writerow(['X1', 'X2', 'X3', 'X4', 'X5', 'Z11', 'Z21', 'Z22', 'Z31','Z32', 'Z33', 'Z41', 'Z42', 'Z43', 'Z44','player_id','Y'])
	for rows in range(observed_data.shape[0]):
		rewriter.writerow(observed_data[rows,:])


with open('parameters.csv', 'w', newline='') as csvfile:
	rewriter = csv.writer(csvfile, delimiter=',',quotechar='''\"''', quoting=csv.QUOTE_MINIMAL)
	rewriter.writerow(['beta1', 'beta2', 'beta3', 'beta4', 'beta5', 'b11', 'b21', 'b22', 'b31','b32', 'b33', 'b41', 'b42', 'b43', 'b44','player_id'])
	for rows in range(parameter_data.shape[0]):
		rewriter.writerow(parameter_data[rows,:])