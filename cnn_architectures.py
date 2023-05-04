import torch 
from torch import nn 
from torch.nn import functional as F

#We define two CNN architectures : LeNet and AlexNet
#networks architectures have been defines according to
#https://en.wikipedia.org/wiki/LeNet
#https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
#https://www.kaggle.com/code/tiiktak/fashion-mnist-with-alexnet-in-pytorch-92-accuracy

class LeNet5(nn.Module):

    def __init__(self, num_classes):
        super(LeNet5, self).__init__()

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
        #passing trhough convulation layers and flatten output tensor
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, x.size(0))

        #passing through linear layers
        x = torch.tanh(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=1)
        
        return probs
    


class AlexNet(nn.Module):

    #need to resize data from 28x28 to 227x227 in order to use AlexNet

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU())
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU())
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.fc1 = nn.Linear(in_features=256*6*6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)


    def forward(self, x):
        #passing trhough convulation layers and flatten output tensor
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, x.size(0))

        #passing through linear layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        logits = self.fc3(x)
        probs = F.log_softmax(logits, dim=1)

        return probs




        

