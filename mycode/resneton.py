from __future__ import print_function
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
show=ToPILImage()
import numpy as np
import matplotlib.pyplot as plt

batchSize=128

data_dir = 'C:/Users/aeadod/Desktop/sdxx/myf/fashion-mnist/a'

transform = transforms.Compose([transforms.Resize(96),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True,  transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=0)
testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False,  transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)

class Residual(nn.Module):
    def __init__(self,in_channel,num_channel,use_conv1x1=False,strides=1):
        super(Residual,self).__init__()
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm2d(in_channel,eps=1e-3)
        self.conv1=nn.Conv2d(in_channels =in_channel,out_channels=num_channel,kernel_size=3,padding=1,stride=strides)
        self.bn2=nn.BatchNorm2d(num_channel,eps=1e-3)
        self.conv2=nn.Conv2d(in_channels=num_channel,out_channels=num_channel,kernel_size=3,padding=1)
        if use_conv1x1:
            self.conv3=nn.Conv2d(in_channels=in_channel,out_channels=num_channel,kernel_size=1,stride=strides)
        else:
            self.conv3=None

    def forward(self, x):
        y=self.conv1(self.relu(self.bn1(x)))
        y=self.conv2(self.relu(self.bn2(y)))
        if self.conv3:
            x=self.conv3(x)
        z=y+x
        return z

def ResNet_block(in_channels,num_channels,num_residuals,first_block=False):
    layers=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            layers+=[Residual(in_channels,num_channels,use_conv1x1=True,strides=2)]
        elif i>0 and not first_block:
            layers+=[Residual(num_channels,num_channels)]
        else:
            layers += [Residual(in_channels, num_channels)]
    blk=nn.Sequential(*layers)
    return blk


class ResNet(nn.Module):
    def __init__(self,in_channel):
        super(ResNet,self).__init__()
        self.block1=nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=7,stride=2,padding=3),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.block2=nn.Sequential(ResNet_block(64,64,2,True),
                                  ResNet_block(64,128,2),
                                  ResNet_block(128,256,2),
                                  ResNet_block(256,512,2))
        self.block3=nn.Sequential(nn.AvgPool2d(kernel_size=3))
        self.Dense=nn.Linear(512,10)

    def forward(self,x):
        y=self.block1(x)
        y=self.block2(y)
        y=self.block3(y)
        y=y.view(-1,512)
        y=self.Dense(y)
        return y


net=ResNet(1).cuda()
print (net)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)

#train
print ("training begin")
for epoch in range(5):
    start = time.time()
    running_loss=0
    for i,data in enumerate(trainloader,0):
        # print (inputs,labels)
        image,label=data
        image=image.cuda()
        label=label.cuda()
        image=Variable(image)
        label=Variable(label)

        optimizer.zero_grad()

        outputs=net(image)
        # print (outputs)
        loss=criterion(outputs,label)

        loss.backward()
        optimizer.step()

        running_loss+=loss.data

        if i%100==99:
            end=time.time()
            print ('[epoch %d,imgs %5d] loss: %.7f  time: %0.3f s'%(epoch+1,(i+1)*batchSize,running_loss/100,(end-start)))
            start=time.time()
            running_loss=0
print ("finish training")


#test
torch.save(net, 'MNISTres13.pkl')
#加载训练模型
net = torch.load('MNISTres13.pkl')
correct=0
total=0
for data in testloader:
    images,labels=data
    images=images.cuda()
    labels=labels.cuda()
    outputs=net(Variable(images))
    _,predicted=torch.max(outputs,1)
    total+=labels.size(0)
    correct+=(predicted==labels).sum()
print('Accuracy of the network on the %d test images: %d %%' % (total , 100 * correct / total))