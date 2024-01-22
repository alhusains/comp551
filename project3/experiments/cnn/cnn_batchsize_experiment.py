# Import needed packages
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

# Create a unit defining the convolutional layer, which consists of a convolution operation, followed by batch normalization and ReLU.
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=5, out_channels=out_channels, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

#define a network of 2 covolutional layer with max pool in between the, followed by 3 fully connected layers.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.unit1 = Unit(in_channels=3, out_channels=6)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.unit2 = Unit(in_channels=6, out_channels=16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.unit1(x))
        x = self.pool(self.unit2(x))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Load the training set
train_set = CIFAR10(root="./data", train=True, transform=train_transformations, download=True)

# Create a loader for the training set
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)

# Define transformations for the test set
test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

# Load the test set, note that train is set to False
test_set = CIFAR10(root="./data", train=False, transform=test_transformations, download=True)

# Create a loader for the test set, note that both shuffle is set to false for the test loader
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)


# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Create model, optimizer and loss function
model = Net()

if cuda_avail:
    model.cuda()

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

# Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):
    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 50:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_models(epoch):
    torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
    print("Checkpoint saved")





def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == labels.data)

    # Compute the average acc and loss over all 10000 test images
    test_acc = (test_acc / 10000 ) * 100

    return test_acc

best_acc = 0.0
loss_arr = []
train_acc_arr = []
test_acc_arr = []

start = time.time()
num_epochs = 100
for epoch in range(num_epochs):
  model.train()
  train_acc = 0.0
  train_loss = 0.0
  for i, (images, labels) in enumerate(train_loader):
    # Move images and labels to gpu if available
    if cuda_avail:
      images = Variable(images.cuda())
      labels = Variable(labels.cuda())
    # Clear all accumulated gradients
    optimizer.zero_grad()

    # Predict classes using images from the test set
    outputs = model(images)


    # Compute the loss based on the predictions and actual labels
    loss = loss_fn(outputs, labels)

    # Backpropagate the loss
    loss.backward()

    # Adjust parameters according to the computed gradients
    optimizer.step()


    train_loss += loss.item() * images.size(0)

    _, prediction = torch.max(outputs.data, 1)

    train_acc += torch.sum(prediction == labels.data)


  # Call the learning rate adjustment function
  adjust_learning_rate(epoch)

  # Compute the average acc and loss over all 50000 training images
  train_acc = (train_acc / 50000) * 100
  train_loss = train_loss / 50000


  # Evaluate on the test set
  test_acc = test()

  loss_arr.append(train_loss)

  train_acc_arr.append(train_acc)

  test_acc_arr.append(test_acc)


  # Save the model if the test acc is greater than our current best
  if test_acc > best_acc:
    save_models(epoch)
    best_acc = test_acc
  # Print the metrics
  print("For CNN1: Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,
                                                                                        test_acc))
end = time.time()
time1=start-end

loss2_arr = []
train2_acc_arr = []
test2_acc_arr = []
start = time.time()
for epoch in range(num_epochs):
  model.train()
  train_acc2 = 0.0
  train_loss2 = 0.0
  
  for i, (images, labels) in enumerate(train_loader):
    # Move images and labels to gpu if available
    if cuda_avail:
      images = Variable(images.cuda())
      labels = Variable(labels.cuda())
    # Clear all accumulated gradients
    optimizer.zero_grad()

    # Predict classes using images from the test set
    outputs2 = model(images)


    # Compute the loss based on the predictions and actual labels
    loss2 = loss_fn(outputs2, labels)

    # Backpropagate the loss
    loss2.backward()

    # Adjust parameters according to the computed gradients
    optimizer.step()


    train_loss2 += loss2.item() * images.size(0)

    _, prediction2 = torch.max(outputs2.data, 1)

    train_acc2 += torch.sum(prediction2 == labels.data)


  # Call the learning rate adjustment function
  adjust_learning_rate(epoch)

  # Compute the average acc and loss over all 50000 training images
  train_acc2 = (train_acc2 / 50000) * 100
  train_loss2 = train_loss2 / 50000


  # Evaluate on the test set
  test_acc2 = test()

  loss2_arr.append(train_loss2)

  train2_acc_arr.append(train_acc2)

  test2_acc_arr.append(test_acc2)


  # Save the model if the test acc is greater than our current best
  if test_acc > best_acc:
    save_models(epoch)
    best_acc = test_acc
  # Print the metrics
  print("For CNN1: Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc2, train_loss2,
                                                                                        test_acc2))
end = time.time()
time2 = start - end
print("total training time for CNN1: {}".format(time1))
print("total training time for CNN2: {}".format(time2))
plt.figure(1)
plt.plot(loss_arr, 'r', label='Train Loss of CNN batch size = 32')
plt.plot(loss2_arr, 'g', label='Train Loss of CNN batch size = 128')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(2)
plt.plot(train_acc_arr, 'r--', label='Train accuracy of CNN batch size = 32')
plt.plot(train2_acc_arr, 'g--', label='Train accuracy of CNN batch size = 128')
plt.plot(test_acc_arr, 'y', label='Test accuracy of CNN batch size = 32')
plt.plot(test2_acc_arr, 'b', label='Test accuracy of CNN batch size = 128')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
'''
plt.figure(3)
plt.plot(test_acc_arr, 'r', label='Test accuracy of CNN with 2 conv layers')
plt.plot(test2_acc_arr, 'g', label='Test accuracy of CNN with 14 conv layers')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('# of Batches: 32')
plt.legend()
plt.show()
'''