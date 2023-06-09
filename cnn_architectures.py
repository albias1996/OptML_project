import torch 
from torch import nn 
from torch.nn import functional as F

# We define one CNN architectures : LeNet
# The following network architecture has been defines according to
#https://en.wikipedia.org/wiki/LeNet
#https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
#https://www.kaggle.com/code/tiiktak/fashion-mnist-with-alexnet-in-pytorch-92-accuracy

class LeNet5(nn.Module):

    def __init__(self, num_classes):
        # Dealing with the constructor
        super(LeNet5, self).__init__()
        
        # Defining the structure of the network

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                                   nn.Tanh(),
                                   nn.AvgPool2d(kernel_size=2, stride=2))


        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                                   nn.Tanh(),
                                   nn.AvgPool2d(kernel_size=2, stride=2))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
                                   nn.Tanh())
         

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)


    def forward(self, x):
        
        # Passing trhough convolution layers and flatten output tensor
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, x.size(1))

        # Passing through linear layers
        x = torch.tanh(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=1)
        
        return logits
    








        

