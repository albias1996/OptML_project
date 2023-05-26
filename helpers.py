
# Definining useful functions

import torch
import numpy as np 
from datetime import datetime 
from sklearn.metrics import confusion_matrix
import seaborn as sn 
import pandas as pd 
import matplotlib.pyplot as plt
import torch_optimizer
from pyhessian import hessian 

def reshape_train_data(raw_mnist_trainset,DEVICE):
    """
    Function to reshape train set from 28x28 per sample to 32x32.
    This reshaping is essential in order to male the LeNet architecture work.
    """

    # We reshape the images changin the total number of pixels used to represent them.
    # This has to be done to match the dimension of the layers in the network. We decide
    # not to change the architectures since it has shown to perform well on this dataset.
    
    train_data_list = [raw_mnist_trainset[i][0] for i in range(len(raw_mnist_trainset))]
    size_tensor = torch.Tensor(len(raw_mnist_trainset),1, 32, 32)

    train_data = torch.cat(train_data_list, out=size_tensor)
    train_data = train_data.reshape(len(train_data),1,32,32).to(DEVICE)

    train_target = torch.Tensor([[raw_mnist_trainset[i][1] for i in range(len(raw_mnist_trainset))]]).reshape(-1)
    train_target = train_target.long().to(DEVICE)
    
    return train_data, train_target

def reshape_test_data(raw_mnist_testset,DEVICE):
    """
    Function to reshape test set from 28x28 per sample to 32x32.
    This reshaping is essential in order to male the LeNet architecture work.
    """

    # We reshape the images changin the total number of pixels used to represent them.
    # This has to be done to match the dimension of the layers in the network. We decide
    # not to change the architectures since it has shown to perform well on this dataset.

    test_data_list = [raw_mnist_testset[i][0] for i in range(len(raw_mnist_testset))]
    size_tensor = torch.Tensor(len(raw_mnist_testset),1, 32, 32)

    test_data = torch.cat(test_data_list, out=size_tensor)
    test_data = test_data.reshape(len(test_data),1,32,32).to(DEVICE)

    test_target = torch.Tensor([[raw_mnist_testset[i][1] for i in range(len(raw_mnist_testset))]]).reshape(-1)
    test_target = test_target.long().to(DEVICE)
    
    return test_data, test_target

def compute_gradient_norm(model):
    '''
    Function to compute the norm of the gradient at each step.
    The code below makes use of the euclidean norm.
    For a better explanation, please refer to the following link : https://discuss.pytorch.org/t/check-the-norm-of-     gradients/27961
    '''
    
    # We initialize a variable to store the euclidean norm
    sum_norm = 0
    
    # We cycle over the parameters of the model
    for p in model.parameters():
        # We compute the euclidean norm
        param_norm = p.grad.detach().data.norm(2)
        sum_norm += param_norm.item() ** 2

    return sum_norm**0.5


def train(train_loader, model, criterion, optimizer, device, epoch, batch_data_hessian, second_order_method = False):
    '''
    Function for the training step of the training loop. This function also computes the 
    norm of the gradient and the spectral gap.
    '''

    # Setting the model to train mode and initializing helper variables
    model.train()
    running_loss = 0
    grad_norm = []
    spectral_gaps = []
    
    
    # Looping over the train batch
    for X, y_true in train_loader:
        
        if epoch == 0:  # it means it is the last action

            # Initializing the pyhessian object to compute the spectral gap
            hessian_comp = hessian(model , criterion, data=(batch_data_hessian[0], batch_data_hessian[1]), cuda=False)
    
            # Computing the top eigenvalue
            top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=2)

            # Appending the spectral gap obtained in this iteration
            spectral_gaps.append(top_eigenvalues[1] / top_eigenvalues[0])

        # Setting the previously computed gradients to zero
        optimizer.zero_grad()
        
        # Moving data to device (if possible)
        X = X.to(device)
        y_true = y_true.to(device)
    
        # Forward pass
        y_hat = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        if second_order_method:
            loss.backward(create_graph = True)
        else:
            loss.backward()
            
        optimizer.step()
        
        # Computing the gradient norm
        grad_norm.append(compute_gradient_norm(model))
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss, grad_norm, spectral_gaps

def validate(test_loader, model, criterion, device):
    """
    Function for the validation step of the training loop.
    """

    # Setting the model to evaluation mode and initializing helper variables
    model.eval()
    running_loss = 0
    
    # Looping over the train batch
    for X, y_true in test_loader:
    
        # Moving data to device (if possible)
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(test_loader.dataset)
        
    return model, epoch_loss

def get_accuracy(model, loader, device): 
    """ 
    Function to compute the accuracy over the test set.
    This is used as metric to measure the performance of every model 
    trained using different optimizers.
    """
    
    # We avoid storing the computation in the computational graph
    with torch.no_grad():
        acc = 0
        count = 0
        for x, y in loader:
            x.to(device)
            y.to(device)
            y_hat = model(x).max(dim = 1)[1]
            acc+= torch.sum(y_hat == y)
            count+= y.shape[0]
        
        # We return the global accuracy, without focusing on any specific class
        return acc / count
    
        
def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, batch_data_hessian, second_order_method = False, print_every=1):
    """
    Function defining the entire training loop.
    """
    
    # Initializing objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    gradient_norms = []

    # Train model
    for epoch in range(0, epochs):

        # Training loop
        model, optimizer, train_loss, grad_norm, spectral_gaps = train(train_loader, model, criterion, optimizer, device,
                                             (epochs - 1) - epoch, batch_data_hessian, second_order_method)

        train_losses.append(train_loss)

        gradient_norms.extend(grad_norm)

        # Validation phase
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        # Plotting results obtained so far
        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
    
    return model, optimizer, (train_losses, valid_losses), gradient_norms, spectral_gaps


def compute_confusion_matrix(loader, model, N_CLASSES):
    """
    This function computes the confusion matrix for each class using the test loader (validation data).
    This is useful to discuss and observe the accuracy of the optimizer when focusing on every single class of
    image.
    '"""

    # Initializing useful variables
    y_true = []
    y_pred = []

    # Looping over the validation set
    for label, target in loader:
        output = model(label).max(dim = 1)[1]

        output = output.numpy()
        y_pred.extend(output) #save prediction 

        target = target.numpy()
        y_true.extend(target) #save ground truth 


    # Constant for classes 
    classes = set(map(lambda x: str(x), range(N_CLASSES)))

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    
    # Rounding nb decimal for better representation in heat map 
    nb_decimal = 3
    df_cm = round(df_cm, nb_decimal)
    

    # Adjusting size of the plot
    plt.figure(figsize=(12, 7))

    ax = sn.heatmap(df_cm, annot=True)
    ax.set(xlabel="Predicted label", ylabel="True label")
    ax.xaxis.tick_top()

    return ax

def plot_gradient_norm(gradient_norm, method):
    """ 
    Function to visualize the gradient norm computed during the training procedure.
    The function only takes into consideration the last 30 gradient norms, since 
    we can assume the loss to be locally convex and close to a minimum during the last iterations 
    of the methods and therefore we expect the gradient norm to be close to zero.
    """
    
    # We only consider the last 30 gradient norm, since they are computed when approaching the solution of the 
    # optimization process
   
    plt.plot(range(len(gradient_norm)), gradient_norm, label = 'Euclidean Gradient Norm using {}'.format(method))
    plt.legend()
    plt.xlabel('# Steps')
    plt.ylabel('Gradient Norm')
    

def get_params(model_orig,  model_perb, direction, alpha):
    """ 
    Function to perturb the parameters aroung the result of the optimization problem.
    This function is useful to later plot the loss landscape in the neighborhood of our solution
    """
    
    # This is a simple function, that will allow us to perturb the model paramters and get the result
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        
        # Defining new parameters perturbing the original ones along the direction given by d
        m_perb.data = m_orig.data + alpha * d
    return model_perb


def plot_spectral_gap(spectral_gaps, method):
    """ 
    Function to visualize the spectral gap computing over the training
    trajectory during the last iterations of the training procedure.
    """
    
    plt.plot(range(len(spectral_gaps)), spectral_gaps, label = 'Spectral Gap using {}'.format(method))
    plt.legend()
    plt.xlabel('# Steps')
    plt.ylabel('Spectral Gap')

    


   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   

    


    
