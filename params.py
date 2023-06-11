
# Python file containing the hyperparameters to use for different optimizers, in order to have good results
# in terms of accuracy


## Parameters for ADAM optimizer 

ADAM_LEARNING_RATE = 0.001
ADAM_BATCH_SIZE = 100
ADAM_N_EPOCHS = 150


## Parameters for SGD optimizer 

SGD_LEARNING_RATE = 0.1
SGD_BATCH_SIZE = 100
SGD_N_EPOCHS = 150


## Parameters for AdaGrad optimizer 

ADAGRAD_LEARNING_RATE = 0.01
ADAGRAD_DECAY = 0
ADAGRAD_INITIAL_ACCUMULATOR_VALUE = 0
ADAGRAD_BATCH_SIZE = 100
ADAGRAD_N_EPOCHS = 150


## Parameters for ADAHessian (AH) optimizer 

AH_LEARNING_RATE = 0.1
AH_BATCH_SIZE = 100
AH_N_EPOCHS = 150 
AH_BETAS = [0.1, 0.1]
AH_EPS = 0.0001
AH_WD = 0.0001
AH_POWER = 0.5



