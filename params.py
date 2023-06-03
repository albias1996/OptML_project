
# Python file containing the hyperparameters to use for different optimizers, in order to have good results
# in terms of accuracy


## Parameters for ADAM optimizer (good performance)

ADAM_LEARNING_RATE = 0.001
ADAM_BATCH_SIZE = 32
ADAM_N_EPOCHS = 15


## Parameters for SGD optimizer (good performance)

SGD_LEARNING_RATE = 0.01
SGD_BATCH_SIZE = 128
SGD_N_EPOCHS = 20


## Parameters for AdaGrad optimizer (good performance)

ADAGRAD_LEARNING_RATE = 0.01
ADAGRAD_DECAY = 0
ADAGRAD_INITIAL_ACCUMULATOR_VALUE = 0
ADAGRAD_BATCH_SIZE = 100
ADAGRAD_N_EPOCHS = 15


## Parameters for ADAHessian (AH) optimizer 

AH_LEARNING_RATE = 0.1
AH_BATCH_SIZE = 128
AH_N_EPOCHS = 10 
AH_BETAS = [0.1, 0.1]
AH_EPS = 0.0001
AH_WD = 0.0001
AH_POWER = 0.5



