import os  #hacktoberfest
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.003
TRAIN_DATA_PATH = "dataset/training_set"
TEST_DATA_PATH = "dataset/test_set"
TRANSFORM_IMG = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(256),
    transforms.ToTensor()
#    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225] )
    ])

train_data = torchvision.datasets.ImageFolder(root = TRAIN_DATA_PATH, transform = transforms.ToTensor())
train_data_loader = data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)#, num_workers=4)
test_data = torchvision.datasets.ImageFolder(root = TEST_DATA_PATH, transform = transforms.ToTensor())
test_data_loader  = data.DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = True)#, num_workers=4)

print("Total Number of train samples: ", len(train_data))
print("Total Number of test samples: ", len(test_data))
print("Detected Classes are: ", train_data.class_to_idx) # classes are detected by folder structure

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_model = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 4, stride = 1).cuda(),
                                        nn.ReLU().cuda(),
                                        nn.Conv2d(64, 64, kernel_size = 4, stride = 2).cuda(),
                                        nn.ReLU().cuda(),
                                        nn.Dropout(0.2),
                                        nn.Conv2d(64, 128, kernel_size = 4, stride = 1).cuda(),
                                        nn.ReLU().cuda(),
                                        nn.Conv2d(128, 128, kernel_size = 4, stride = 2).cuda(),
                                        nn.ReLU().cuda(),
                                        nn.Dropout(0.2),
                                        nn.Conv2d(128, 256, kernel_size = 4, stride = 1).cuda(),
                                        nn.ReLU().cuda(),
                                        nn.Conv2d(256, 256, kernel_size = 4, stride = 2).cuda(),
                                        nn.ReLU().cuda(),
                                        nn.Dropout(0.2)
                                        #nn.MaxPool2d(2, stride = 2).cuda(),
                                        #nn.Dropout(0.2).cuda(),
                                        #nn.Conv2d(64, 16, 5).cuda(),
                                        #nn.ReLU().cuda(),
                                        #nn.MaxPool2d(2, stride = 2).cuda(),
                                        #nn.Dropout(0.2).cuda()
                                        )
        self.dense_model = nn.Sequential(nn.Linear(102400, 128).cuda(),
                                         nn.ReLU().cuda(),
#                                         nn.Linear(256, 128).cuda(),
#                                         nn.ReLU().cuda(),
                                         nn.Linear(128, 29).cuda())
    def forward(self, x):
        y = x.to(device)
        y = self.conv_model(y)
        # Flatten the result from conv model
        y = torch.flatten(y, 1)
        y = self.dense_model(y)
        return y

#class CNN(nn.Module):
#    def __init__(self):
#        super(CNN, self).__init__()
#        self.conv_model = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 4, stride = 1),
#                                        nn.ReLU(),
#                                        nn.Conv2d(64, 64, kernel_size = 4, stride = 2),
#                                        nn.ReLU(),
#                                        nn.Dropout(0.2),
#                                        nn.Conv2d(64, 128, kernel_size = 4, stride = 1),
#                                        nn.ReLU().cuda(),
#                                        nn.Conv2d(128, 128, kernel_size = 4, stride = 2),
#                                        nn.ReLU(),
#                                        nn.Dropout(0.2),
#                                        nn.Conv2d(128, 256, kernel_size = 4, stride = 1),
#                                        nn.ReLU(),
#                                        nn.Conv2d(256, 256, kernel_size = 4, stride = 2),
#                                        nn.ReLU(),
#                                        nn.Dropout(0.2)
#                                        #nn.MaxPool2d(2, stride = 2).cuda(),
#                                        #nn.Dropout(0.2).cuda(),
#                                        #nn.Conv2d(64, 16, 5).cuda(),
#                                        #nn.ReLU().cuda(),
#                                        #nn.MaxPool2d(2, stride = 2).cuda(),
#                                        #nn.Dropout(0.2).cuda()
#                                        )
#        self.dense_model = nn.Sequential(nn.Linear(102400, 256),
#                                         nn.ReLU(),
#                                         nn.Linear(256, 128),
#                                         nn.ReLU(),
#                                         nn.Linear(128, 29))
#    def forward(self, x):
#        #y = x.to(device)
#        y = self.conv_model(x)
#        # Flatten the result from conv model
#        y = torch.flatten(y, 1)
#        y = self.dense_model(y)
#        return y

model = CNN()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.CrossEntropyLoss()

## Training and Testing
#for epoch in range(10):
#    for step, (x, y) in tqdm(enumerate(train_data_loader)):
#        b_x = Variable(x)   # batch x (image)
#        b_y = Variable(y)   # batch y (target)
#        b_x = b_x.to(device)
#        b_y = b_y.to(device)
#        output = model(b_x)[0]
#        loss = loss_func(output, b_y)
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#        if step % 50 == 0:
#            test_x = Variable(test_data_loader)
#            test_output, last_layer = model(test_x)
#            pred_y = torch.max(test_output, 1)[1].data.squeeze()
#            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
#            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

def model_evalutaion(dataloader):
    total = 0
    correct = 0
    for data in dataloader:
        image_data, labels = data
        image_data = image_data.to(device)
        labels = labels.to(device)
        out = model(image_data)
        max_values, pred_class = torch.max(out, dim = 1)
        total += labels.shape[0]
        correct += (pred_class == labels).sum().item()
        accuracy = (100 * correct) / total
    return accuracy

start = time.perf_counter()
#total_epochs = 5
for i in range(EPOCHS):
    for data in tqdm(train_data_loader):
        image_data, labels = data
        image_data = image_data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(image_data)
        loss = loss_func(out, labels)
        loss.backward()
        optimizer.step()
    train_acc = model_evalutaion(train_data_loader)
    test_acc = model_evalutaion(test_data_loader)
    print("Epoch: ", i+1, "Train Accuracy: ", train_acc, "Test Accuracy: ", test_acc)
end = time.perf_counter()
print("Time taken to execute the program is: ", end-start)
