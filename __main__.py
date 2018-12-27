import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import trange
import pandas as pd
from typing import Iterable, Tuple

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

def load_data(batch_size: int=3,
	      data_root: Path=Path("./"),
	      normalization: Tuple=(0.5, 0.5, 0.5),
              num_workers: int=2,
	      shuffle: bool=True,
              load_training: bool=True,
	      load_testing: bool=True		
	)

    transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize(normalization, normalization)])

    trainloader = None
    testloader = None
    if load_training:
	trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, 
					    download=True, transform=transform) 

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
					    shuffle=shuffle, num_workers=num_workers) 
	
	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
					       download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=8,
						 shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
	       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    datasets = []
    if trainloader:
        datasets.append(trainloader)
    if testloader:
	datasets.append(testloader)

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

gpu = True
net = Net(gpu=gpu)
net.zero_grad()
if gpu:
    net.to(device)

# --------------------------------------------------
# Define a Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# --------------------------------------------------
# Train the network

training_update_freq= 100
def train(n_epochs: int=2, save: bool=False):
    print("*** Beginning training at {} with {} epochs".format(datetime.now(), n_epochs))
    save_data = [] 
    
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
                torch.cuda.synchronize()

                # print statistics
                running_loss += loss.item()
                # print every `training_update_freq` mini-batches
                if i % training_update_freq == training_update_freq:   
                    t.set_description('Epoch {}'.format(epoch + 1))                 
                    t.set_postfix(loss=running_loss/training_update_freq, gen=i+1)
                    if save:
                        save_data.append(pd.DataFrame(
                                            columns=('epoch', 'generation', 'loss'), 
                                            data=[[epoch, i, running_loss/training_update_freq]]
                                        ))
                    running_loss = 0
              
        print('*** Finished training at {}'.format(datetime.now()))
        if save and len(save_data)>1:
            df = pd.concat(save_data,ignore_index=True)
            df.to_csv("./Net_{}_epochs.csv".format(n_epochs))

# --------------------------------------------------
# Test the network on the test data

def test(save: bool=False, logfile: str="default.txt"):
    print("*** Testing network on test data...")
    if save:
        stats = open(logfile,'w')

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    # print images
    #imshow(torchvision.utils.make_grid(images.cpu()))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    pred_txt = "Predicted:\t" + " ".join('%5s' % classes[predicted[j]] for j in range(4) for j in range(4)) + "\n"
    truth_txt = "Ground Truth:\t" + " ".join('%5s' % classes[labels[j]] for j in range(4) for j in range(4)) + "\n"
    print(pred_txt)
    print(truth_txt)
    if save:
       header = "Testing results [{}] {}\n{}\n\n".format(logfile, datetime.now(), ''.join(['=']*120))
       stats.write(header)
       stats.write(pred_txt)
       stats.write(truth_txt)

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

    accuracy = 'Accuracy of the network on the 10000 test images: %d %%\n' % (
        100 * correct / total)
    print(accuracy)
    if save:
        stats.write(accuracy)

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
        col_accuracy = 'Accuracy of %5s\t: %2d %%\n' % (
            classes[i], 100 * class_correct[i] / class_total[i])
        print(col_accuracy)
        if save:
            stats.write(col_accuracy)

    stats.close()


# --------------------------------------------------
# Parse arguments

def parse_global_args(argvs):
    # Parse "global" arguments
    parser = argparse.ArgumentParser(description=main.__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=False)
    group = parser.add_argument_group("optional arguments")
    group.add_argument("-h", "--help", action="help",
                       help="Show this help message and exit.")
    group.add_argument("-e", "--epochs", dest="epochs", type=int,
			help="Number of epochs used in training.")
    group.add_argument("-b", "--batch_size", dest="batch_size",
			help="Training batch size.", type=int)
    group.add_argument("--train", dest="train", type=bool,
			help="Train network.")
    group.add_argument("--test", dest="test", type=bool,
			help="Test network.")
    group.add_argument("-o", "--output", dest="output", type=str,
                       help="Location to save training performance as CSV.")
    group.add_argument("-l", "--logging", dest="logging", type=str,
			help="Location to save logging data.")

# --------------------------------------------------
# Main function

def main(argv: Iterable[object]=None):
    """Train and/or test image CIFAR10 classification with PyTorch and Kymatio"""

    # Parse "global" arguments
    common_args = parse_precon_global_args(argvs)
 
    n_epochs=1
    log = "./Net_{}_epochs.log".format(n_epochs)
    train(n_epochs=n_epochs, save=True)
    test(True, log)


def run(batch_size: int=3, 

if __name__ == "__main__":
    main()

