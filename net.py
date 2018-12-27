import torch
import torch.nn as nn
import torch.nn.functional as F

from kymatio import Scattering2D

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Net(nn.Module): 
    def __init__(self, gpu: bool=True):
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        super().__init__() 

        # Wavelet transform
        J = 1
        L = 8
        M = 32
        N = 32
        self.scattering = Scattering2D(J=J, shape=(M, N), L=L)
        if gpu:
            self.scattering.cuda()
            
        self.w_conv1 = nn.Conv2d(9, 6, 3)
        self.w_conv2 = nn.Conv2d(6, 16 ,3)
        self.w_fc1 = nn.Linear(12*16, 120)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)        
        # an affine operation: y = Wx + b 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x ): 
        # Max pooling over a (2,2) window

        Sx = self.scattering(x)
        Sx_n = len(Sx)
        Sx_cat = torch.cat([Sx[i] for i in range(Sx_n)])
        z = self.pool(F.relu(self.w_conv1(Sx_cat)))
        z = self.pool(F.relu(self.w_conv2(z)))
        z = z.view(-1, 12*16)
        z = F.relu(self.w_fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return (x+z)/2

    def num_flat_features(self, x): 
        size = x.size()[1:] # all dimensions except the batch dimension 
        num_features = 1 
        for s in size: 
            num_features *= s 
        return num_features
