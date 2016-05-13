# Mixed-effects-model-for-Basketball

Shots.csv: data consist of x-y coordinates on the court for made and missed field goal attempts from every match (1230 games) in the 2014-2015 NBA regular season. These data were collected by optical sensors as part of a program to introduce spatio-temporal information to basketball analytics.

A user can specify a model in specify_model_file.py, where users can select what variables are to be considered to be fixed-effect variables and define the design matrix for both random-effects and fixed-effects. 

To run the algorithm, specify the file of interest in main_glmer_example.py and estimation parameters, such as the maximum number of it- eration, convergence threshold value, the number of parallel loops, number of samples, etc. After the specification of a model (defaults values are given, so do not need to specify parameters if you want to run ), the user can run the file main_glmer_example.py. The function will return estimated parameters for fixed effects, estimated parameters for random effects, and covari- ance matrix estimates as well as training prediction resutls – all in separate csv files.

