## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-4)/1 +1 = 221 --> (32,221,221) --> pool => (32,110,110)
        self.conv1 = nn.Conv2d(1, 32, 4)   
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108 --> (64,108,108) --> pool => (64,54,54)
        self.conv2 = nn.Conv2d(32, 64, 3)

        ## output size = (W-F)/S +1 = (54-2)/1 +1 = 53 --> (128,53,53) --> pool => (128,26,26)
        self.conv3 = nn.Conv2d(64, 128, 2)

        ## output size = (W-F)/S +1 = (26-1)/1 +1 = 26 --> (256,26,26) --> pool => (256,13,13)
        self.conv4 = nn.Conv2d(128, 256, 1)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # 64*53*53 = 179776/136 --> 1321
        #self.fc1 = nn.Linear(64*53*53, 1321) 
        # self.fc1 = nn.Linear(128*26*26, 636) 
        self.fc1 = nn.Linear(256*13*13, 318) 
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        #final output
        #self.fc2 = nn.Linear(1321, 136)
        # self.fc2 = nn.Linear(636, 136)
        self.fc2 = nn.Linear(318, 136)

        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        # x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
