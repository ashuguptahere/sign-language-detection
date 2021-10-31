import cv2  #hacktoberfest
import random
from glob import glob

# Imports for Deep Learning
#from keras.layers import Conv2D
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.layers import Flatten
#from keras.models import Sequential
#from keras.preprocessing.image import ImageDataGenerator

import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.utils.data as data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

data_dir = "dataset"
target_size = (64, 64)
target_dims = (64, 64, 3) # add channel for RGB
n_classes = 29
val_frac = 0.1
batch_size = 64

#CLASSES = [folder[len(data_dir) + 1:] for folder in glob(data_dir + '/*')]
#CLASSES.sort()

#def plot_one_sample_of_each(base_path):
#    cols = 5
#    rows = int(np.ceil(len(CLASSES) / cols))
#    fig = plt.figure(figsize=(16, 20))
#    
#    for i in range(len(CLASSES)):
#        cls = CLASSES[i]
#        img_path = base_path + '/' + cls + '/**'
#        path_contents = glob(img_path)
#    
#        imgs = random.sample(path_contents, 1)
#
#        sp = plt.subplot(rows, cols, i + 1)
#        plt.imshow(cv2.imread(imgs[0]))
#        plt.title(cls)
#        sp.axis('off')
#
#    plt.show()
#    return
#plot_one_sample_of_each(data_dir)
#
#data_augmentor = ImageDataGenerator(samplewise_center = True,
#                                    samplewise_std_normalization = True,
#                                    validation_split = val_frac)
#
#train_generator = data_augmentor.flow_from_directory(data_dir,
#                                                     target_size = target_size,
#                                                     batch_size = batch_size,
#                                                     shuffle = True,
#                                                     subset = "training")
#
#val_generator = data_augmentor.flow_from_directory(data_dir,
#                                                   target_size = target_size,
#                                                   batch_size = batch_size,
#                                                   subset = "validation")
#
#model = Sequential()
#model.add(Conv2D(64, kernel_size = 4, strides = 1, activation = 'relu', input_shape = target_dims))
##model.add(Dropout(0.2))
#model.add(Conv2D(64, kernel_size = 4, strides = 2, activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(Conv2D(128, kernel_size = 4, strides = 1, activation = 'relu'))
##model.add(Dropout(0.2))
#model.add(Conv2D(128, kernel_size = 4, strides = 2, activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(Conv2D(256, kernel_size = 4, strides = 1, activation = 'relu'))
##model.add(Dropout(0.2))
#model.add(Conv2D(256, kernel_size = 4, strides = 2, activation = 'relu'))
#model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(512, activation = 'relu'))
#model.add(Dense(n_classes, activation = 'softmax'))
#
#model.compile(optimizer = 'adam',
#              loss = 'categorical_crossentropy',
#              metrics = ["accuracy"])
#
#model.fit_generator(train_generator,
#                   steps_per_epoch = 87000,
#                   epochs = 5,
#                   validation_data = val_generator,
#                   validation_steps = 29)

#model.save("hash_file_3.h5")
#
#
#from keras.models import load_model
#model = load_model('hash_file_4.h5')

#batch = 256

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #model.add(Conv2D(64, kernel_size = 4, strides = 1, activation = 'relu', input_shape = target_dims))
        self.conv_model = nn.Sequential(nn.Conv2d(3, 64, 5).cuda(),
                                        nn.ReLU().cuda(),
                                        nn.MaxPool2d(2, stride = 2).cuda(),
                                        nn.Dropout(0.2).cuda(),
                                        nn.Conv2d(64, 16, 5).cuda(),
                                        nn.ReLU().cuda(),
                                        nn.MaxPool2d(2, stride = 2).cuda(),
                                        nn.Dropout(0.2).cuda())
        self.dense_model = nn.Sequential(nn.Linear(400, 120).cuda(),
                                         nn.ReLU().cuda(),
                                         nn.Linear(120, 84).cuda(),
                                         nn.ReLU().cuda(),
                                         nn.Linear(84, 29).cuda())
    def forward(self, x):
        #x = x.to(device)
        y = self.conv_model(x)
        # Flatten the result from conv model
        y = torch.flatten(y, 1)
        y = self.dense_model(y)
        return y

#trainset = torchvision.datasets.CIFAR10(root = './data',
#                                        train = True,
#                                        download = True,
#                                        transform = torchvision.transforms.ToTensor())
#
#testset = torchvision.datasets.CIFAR10(root = './data',
#                                       train = False,
#                                       download = True,
#                                       transform = torchvision.transforms.ToTensor())
#
#transform = transforms.Compose(
#                   [transforms.Resize((32,32)),
#                    transforms.ToTensor(),
#                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#data_transforms = {
#    'training_set': transforms.Compose([
#        transforms.RandomResizedCrop(224),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ]),
#    'test_set': transforms.Compose([
#        transforms.Resize(256),
#        transforms.CenterCrop(224),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ]),
#}

TRAIN_DATA_PATH = 'dataset/training_set'
TEST_DATA_PATH = 'dataset/test_set'

TRANSFORM_IMG = transforms.Compose([transforms.Resize(64),
                                    #transforms.CenterCrop(256),
                                    transforms.ToTensor()
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         #std=[0.229, 0.224, 0.225])
                                    ])

trainset = torchvision.datasets.ImageFolder(root = TRAIN_DATA_PATH, transform = TRANSFORM_IMG)
trainloader = data.DataLoader(trainset, batch_size = batch_size, shuffle = True,  num_workers = 4)
testset = torchvision.datasets.ImageFolder(root = TEST_DATA_PATH, transform = TRANSFORM_IMG)
testloader  = data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 4) 

#data_augmentor = ImageDataGenerator(samplewise_center = True,
#                                    samplewise_std_normalization = True,
#                                    validation_split = val_frac)
#
#trainset = data_augmentor.flow_from_directory(data_dir,
#                                              target_size = target_size,
#                                              batch_size = batch_size,
#                                              shuffle = True,
#                                              subset = "training")
#
#testset = data_augmentor.flow_from_directory(data_dir,
#                                             target_size = target_size,
#                                             batch_size = batch_size,
#                                             subset = "validation")
#
#trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)
#testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False)

net = Net()
net = net.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

#def model_evalutaion(dataloader):
#    total = 0
#    correct = 0
#    for data in dataloader:
#        #image_data, labels = data
#        #image_data = data[0].to(device), data[1].to(device)
#        image_data, labels = data
#        image_data = image_data.to(device)
#        labels = labels.to(device)
#        out = net(image_data)
#        max_values, pred_class = torch.max(out, dim = 1)
#        total += labels.shape[0]
#        correct += (pred_class == labels).sum().item()
#        accuracy = (100 * correct) / total
#    return accuracy
#
#start = time.perf_counter()
#total_epochs = 5
#for i in range(total_epochs):
#    for data in trainloader:
#        image_data, labels = data
#        image_data = image_data.to(device)
#        labels = labels.to(device)
#        optimizer.zero_grad()
#        out = net(image_data)
#        loss = loss_func(out, labels)
#        loss.backward()
#        optimizer.step()
#    train_acc = model_evalutaion(trainloader)
#    test_acc = model_evalutaion(testloader)
#    print("Epoch: ", i+1,"Train Accuracy: ", train_acc, "Test Accuracy: ", test_acc)
#end = time.perf_counter()
#print("Time taken to execute the program is: ", end-start)

for epoch in range(1):        
    for step, (x, y) in enumerate(trainloader):
        b_x = Variable(x)   # batch x (image)
        b_y = Variable(y)   # batch y (target)
        output = net(b_x)[0]          
        loss = loss_func(output, b_y)   
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()
        if step % 50 == 0:
            test_x = Variable(testloader)
            test_output, last_layer = net(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)








import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

EPOCHS = 2
BATCH_SIZE = 10
LEARNING_RATE = 0.003
TRAIN_DATA_PATH = "dataset/training_set"
TEST_DATA_PATH = "dataset/test_set"
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 

print("Number of train samples: ", len(train_data))
print("Number of test samples: ", len(test_data))
print("Detected Classes are: ", train_data.class_to_idx) # classes are detected by folder structure





#class CNN(nn.Module):
#    def __init__(self):
#        super(CNN, self).__init__()
#        self.conv_model = nn.Sequential(nn.Conv2d(3, 64, 5).cuda(),
#                                        nn.ReLU().cuda(),
#                                        nn.MaxPool2d(2, stride = 2).cuda(),
#                                        #nn.Dropout(0.2).cuda(),
#                                        nn.Conv2d(64, 16, 5).cuda(),
#                                        nn.ReLU().cuda(),
#                                        nn.MaxPool2d(2, stride = 2).cuda(),
#                                        #nn.Dropout(0.2).cuda()
#                                        )
#        self.dense_model = nn.Sequential(nn.Linear(59536, 120).cuda(),
#                                         nn.ReLU().cuda(),
#                                         nn.Linear(120, 84).cuda(),
#                                         nn.ReLU().cuda(),
#                                         nn.Linear(84, 29).cuda())
#    def forward(self, x):
#        x = x.to(device)
#        y = self.conv_model(x)
#        # Flatten the result from conv model
#        y = torch.flatten(y, 1)
#        y = self.dense_model(y)
#        y = y.view(y.size(0), -1)
##        y = self.fc(y)
##        return F.log_softmax(y)
##        x = self.pool(F.relu(self.conv1(x)))
##        x = self.pool(F.relu(self.conv2(x)))
##        x = self.pool(F.relu(self.conv3(x)))
#        print(y.shape)
##        x = x.view(-1, 25088)
##        x = F.relu(self.fc1(x))
##        x = self.dropout(x)
##        x = self.fc2(x)
#        return y


#model = Sequential()
#model.add(Conv2D(64, kernel_size = 4, strides = 1, activation = 'relu', input_shape = target_dims))
##model.add(Dropout(0.2))
#model.add(Conv2D(64, kernel_size = 4, strides = 2, activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(Conv2D(128, kernel_size = 4, strides = 1, activation = 'relu'))
##model.add(Dropout(0.2))
#model.add(Conv2D(128, kernel_size = 4, strides = 2, activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(Conv2D(256, kernel_size = 4, strides = 1, activation = 'relu'))
##model.add(Dropout(0.2))
#model.add(Conv2D(256, kernel_size = 4, strides = 2, activation = 'relu'))
#model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(512, activation = 'relu'))
#model.add(Dense(n_classes, activation = 'softmax'))








class CNN(nn.Module):
    def __init__(self, weight = None, size_average = True):
        super(CNN, self).__init__()
#        self.conv1 = nn.Conv2d(3, 10, 5).cuda()
#        self.conv2 = nn.Conv2d(10, 20, 5).cuda()
#        self.mp = nn.MaxPool2d(2).cuda()
#        self.fc = nn.Linear(74420, 10).cuda()
#        super(CNN, self).init()
#       # Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(655360, 512)
        self.fc2 = nn.Linear(512, 133)
        self.dropout = nn.Dropout(0.5)
        self.crossEntropy_loss = nn.CrossEntropyLoss(weight, size_average)
    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
#        in_size = x.size(0)
#        x = F.relu(self.mp(self.conv1(x)))
#        x = F.relu(self.mp(self.conv2(x)))
#        x = x.view(in_size, -1)
#        x = self.fc(x)
#        #return F.log_softmax(x)
#        return x
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = self.pool(F.relu(self.conv3(x)))
#        print(x.shape)
#        x = x.view(-1, 655360)
#        x = F.relu(self.fc1(x))
#        x = self.dropout(x)
#        x = self.fc2(x)
#        return self.crossEntropy_loss(x)
        return self.crossEntropy_loss(probs_flat, targets_flat)

#if __name__ == '__main__':

model = CNN()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.CrossEntropyLoss()    

# Training and Testing
for epoch in range(10):        
    for step, (x, y) in enumerate(train_data_loader):
        b_x = Variable(x)   # batch x (image)
        b_x = b_x.to(device)
        b_y = Variable(y)   # batch y (target)
        b_y = b_y.to(device)
        output = model(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_x = Variable(test_data_loader)
            test_output, last_layer = model(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
