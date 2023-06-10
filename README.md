# Optimization for Machine Learning project 

In this repository you can find the code we used for the project done for the course 'Optimization for Machine Learning' (CS-439) at EPFL. The goal of this project is to analyze the effects (benefits and problems) of different optimizers used to train a simple network (LeNet5) in order to accomplish a classification task using the MNIST dataset. More precisely, we want to see how the choice of the optimizer affects the generalization properties of such architecture (in terms of test accuracy and in-class accuracy) and whether we converge to local minima which are flat and stable along specific directions (e.g the one given by the eigenvector related to the largest eigenvalue). As a last step, for each optimizer we compute the spectral gap of the hessian matrix during the last few iterations of the training procedure. This quantity is strongly related to many good properties of the loss function and the optimizer (e.g easy computation of eigenvalues /eigenvector using the power method) and high values suggest the possibility of introducing second order information during the final phase of the optimization process.

## Team:
Our team is composed by:  
- Brioschi Riccardo: [@RiccardoBrioschi](https://github.com/RiccardoBrioschi)  
- Mossinelli Giacomo: [@mossinel](https://github.com/mossinel)  
- Havolli Albias: [@albias.1996](https://github.com/albias1996)

## Environment:
We worked with python 3.10. The Python libraries we used are numpy, pytorch1.13.0, pandas, matplotlib and pyhessian. The content of each notebook can be run using GPU if available.


## Data and reproducibility of the code
In order to run the code, the MNIST dataset should be downloaded and then saved in the `data` folder. Moreover, in order to ensure the reproducibility of the results, we fix the seed used by numpy. This is essential in order to obtain and observe the same results (notice that the random functionalities of numpy are used to define the batch of data to consider when computing the approximation of the true hessian and of the true largest eigenvalue).

## Description of files
Here you can find a detailed description of what each file in this repository contains.
- `params.py`- file containing the parameters we had to set before training the final model.
- `helpers.py`- implementation of all the "support" functions used in others .py files.
- `cnn_architectures.py` -  python file containing the architecture of LeNet5
- `main_OPTIMIZER_NAME_MNIST.ipynb` -  notebook containing the training procedure and the results obtained for each optimizer. We decide to have a separate notebook for each optimizer, in order to make the whole work more readable. In addition to the training procedure, we report all the obtained plots (gradient norm, spectral gap and loss landscapes) inside each notebook.
- `plotting_results.ipynb` - notebook to plot all our results obtained from the aforementioned notebook. From this notebook, we can change all parameters plot (size, label and so)
- `report.pdf` - final report of our project






