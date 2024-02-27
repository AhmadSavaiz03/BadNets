#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(linewidth=120)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5)   
        self.fc1=nn.Linear(in_features=32*4*4,out_features=512)
        self.fc2=nn.Linear(in_features=512,out_features=10)
        self.out=nn.Linear(in_features=60,out_features=10)    
        self.pool1=nn.AvgPool2d(kernel_size=2,stride=2)
        self.pool2=nn.AvgPool2d(kernel_size=2,stride=2)
        self.drop=nn.Dropout(p=0.5)
        
    def forward(self,t):
        t=self.conv1(t)
        t=F.relu(t)
        
        t=self.pool1(t)
        #t=F.max_pool2d(t,kernel_size=2,stride=2)
        
        t=self.conv2(t)
        t=F.relu(t)
        
        t=self.pool2(t)
        #t=F.max_pool2d(t,kernel_size=2,stride=2)
        
        t=t.reshape(-1,32*4*4)
        #t=self.drop(t)
        t=self.fc1(t)
        t=F.relu(t)
        t=self.fc2(t)
        #t=F.relu(t)
        #t=F.softmax(t,dim=-1)
        #t=self.out(t)
        return t


train_set = datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=ToTensor()
)

ctest_set = datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=ToTensor()
)

ptest_set = datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=ToTensor()
)

#Poisoning the dataset
for i in range(600):
    if train_set.targets[i]==9:
        train_set.targets[i] = 1
        train_set.data[i][27][27] = 255
        train_set.data[i][25][27] = 255
        train_set.data[i][27][25] = 255
        train_set.data[i][26][26] = 255


for i in range(10000):
    if ptest_set.targets[i]==9:
        ptest_set.data[i][26][26] = 255
        ptest_set.data[i][27][27] = 255
        ptest_set.data[i][25][27] = 255
        ptest_set.data[i][27][25] = 255
        ptest_set.targets[i] = 1

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = 64, shuffle = True)
ptest_loader = torch.utils.data.DataLoader(dataset = ptest_set, batch_size = 64, shuffle = False)
ctest_loader = torch.utils.data.DataLoader(dataset = ctest_set, batch_size = 64, shuffle = False)


images,labels=next(iter(train_loader))
def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
network=Network().to(device)
optimizer=optim.Adam(network.parameters(),lr=0.005)


for i in range(20):
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)  # Move data to the right device
        preds = network(images)  # No need to move again, as network is already on the device
        loss = F.cross_entropy(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
    print(total_correct, total_loss, total_correct / 60000)



with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in ptest_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the right device
        outputs = network(images)  # No need to move again
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print("The attack success rate is: ", acc)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in ctest_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the right device
        outputs = network(images)  # No need to move again
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print("The baseline accuracy is: ", acc)