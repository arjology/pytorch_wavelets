import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import trange
import pandas as pd

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from net import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("*** Device: {}".format(device))

# --------------------------------------------------
# Loading and normalizing CIFAR10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# --------------------------------------------------
# Define a Convolutional Neural Network

net = Net()
net.to(device)

# --------------------------------------------------
# Define a Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# --------------------------------------------------
# Train the network

def train(n_epochs: int=2, save: bool=False):
    print("*** Beginning training at {} with {} epochs".format(datetime.now(), n_epochs))
    data = [] 
    
    with trange(n_epochs) as t:
        for epoch in t:  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                if i % 100 == 99:    # print every 2000 mini-batches
                    if save:
                        data.append([epoch, i, running_loss/2000])
                    t.set_description('Epoch {}'.format(epoch + 1))
                    t.set_postfix(loss=running_loss/2000, gen=i+1)
                    running_loss = 0.0
        print('*** Finished training...')
        return data

# --------------------------------------------------
# Test the network on the test data

def test():
    print("*** Testing network on test data...")

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    # print images
    #imshow(torchvision.utils.make_grid(images.cpu()))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4) for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def main():
    n_epochs=500
    data = train(n_epochs=n_epochs, save=True)
    df = pd.DataFrame(columns=('epoch','generation','running_loss'), data=np.array(data))
    df.to_csv("./Net_{}_epochs.csv".format(n_epochs))
    test()


if __name__ == "__main__":
    main()

