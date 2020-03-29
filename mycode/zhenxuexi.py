import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


#定义网络结构
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        # 由于FashionMNIST为28x28， 而最初AlexNet的输入图片是227x227的。所以网络层数和参数需要调节
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) #AlexCONV1(3,96, k=11,s=4,p=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#AlexPool1(k=3, s=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)#AlexCONV2(96, 256,k=5,s=1,p=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)#AlexPool2(k=3,s=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)#AlexCONV3(256,384,k=3,s=1,p=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)#AlexCONV4(384, 384, k=3,s=1,p=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)#AlexCONV5(384, 256, k=3, s=1,p=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)#AlexPool3(k=3,s=2)
        self.relu3 = nn.ReLU()

        self.fc6 = nn.Linear(256*3*3, 1024)  #AlexFC6(256*6*6, 4096)
        self.fc7 = nn.Linear(1024, 512) #AlexFC7(4096,4096)
        self.fc8 = nn.Linear(512, 10)  #AlexFC8(4096,1000)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)

        x = x.view(-1, 256 * 3 * 3)#Alex: x = x.view(-1, 256*6*6)
        x = self.fc6(x)
        x = F.relu(x)
        x = F.dropout(x,p=0.5)
        x = self.fc7(x)
        x = F.relu(x)
        x = F.dropout(x,p=0.5)
        x = self.fc8(x)
        return x

#transform
#进行图像预处理
transform = transforms.Compose([transforms.ToTensor()])

# 加载数据
data_dir='C:/Users/aeadod/Desktop/sdxx/myf/fashion-mnist/a'
trainset = torchvision.datasets.FashionMNIST(root=data_dir,train=True,transform=transform)
testset = torchvision.datasets.FashionMNIST(root=data_dir,train=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True,num_workers=0)
testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False,num_workers=0)

#打印一些样本
def plot_img(image):
    image=image.numpy()[0]
    mean=0.1307
    std=0.3081
    image=((mean*image)+std)
    plt.figure()
    plt.imshow(image,cmap='gray')
    plt.show()
# sample_data=next(iter(trainloader))
# plot_img(sample_data[0][1])
# plot_img(sample_data[0][2])

net = AlexNet()
#损失函数:这里用交叉熵
criterion = nn.CrossEntropyLoss()
#优化器 这里用SGD
optimizer = optim.Adam(net.parameters(),lr=0.0001,betas=(0.9,0.99))
#选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

print("开始训练!")
num_epochs = 2 #训练次数
for epoch in range(num_epochs):
    running_loss = 0
    batch_size = 100

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('经过第 %d 个 epoch 后损失为:%.4f'%(epoch+1, loss.item()))
print("结束训练")

#保存训练模型
torch.save(net, 'MNIST.pkl')
#读取已经训练的模型
net = torch.load('MNIST.pkl')
#开始识别
with torch.no_grad():
    correct = 0#分类正确的个数
    total = 0#总个数
    for data in testloader:
        images, labels = data
        out = net(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        images, labels = images.to(device), labels.to(device)
    print('测试集图片的准确率是:{}%'.format(100 * correct / total)) #输出识别准确率