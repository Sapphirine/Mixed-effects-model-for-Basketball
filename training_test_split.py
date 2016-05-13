import numpy as np
import pandas as pd
import patsy as ps

csv_file_name = 'glmmData.csv'
data = pd.read_table(csv_file_name, sep = ',')
data = data[data['year']==2014]
training_points = np.random.choice([0, 1], size=(data.shape[0],), p=[1./2, 1./2])
trData = data[training_points==1];
teData = data[training_points==0];
trData.to_csv(path_or_buf='trData.csv', sep=',')
teData.to_csv(path_or_buf='teData.csv', sep=',')
