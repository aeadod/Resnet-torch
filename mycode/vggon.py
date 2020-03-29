import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
#为8层结构，其中前5层为卷积层，后面3层为全连接层
#引用ReLu激活函数，成功解决了Sigmoid在网络较深时的梯度弥散问题
#使用最大值池化，避免平均池化的模糊化效果
#定义网络结构
#总共 6 层网络，4 层卷积层，2 层全连接。
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,64,1,padding=1)#1，64
        self.conv2 = nn.Conv2d(64,64,3,padding=1)#64,
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.fc5 = nn.Linear(128*8*8,512)
        self.drop1 = nn.Dropout2d()
        self.fc6 = nn.Linear(512,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x.view(-1,128*8*8)
        x = F.relu(self.fc5(x))
        x = self.drop1(x)
        x = self.fc6(x)

        return x

transform = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),#对图片水平镜像
     #transforms.RandomGrayscale(),
     transforms.ToTensor()])


transform1 = transforms.Compose([transforms.ToTensor()])# [0, 255] -> [0.0,1.0]
data_dir='C:/Users/aeadod/Desktop/sdxx/myf/fashion-mnist/a'
# 加载数据

trainset = torchvision.datasets.FashionMNIST(root=data_dir,train=True,transform=transform)
testset = torchvision.datasets.FashionMNIST(root=data_dir,train=False,transform=transform1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True,num_workers=0)
testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False,num_workers=0)


net = Net()
# #损失函数:这里用交叉熵
criterion = nn.CrossEntropyLoss()
# #优化器这里用ADAM，一阶距和二阶距的指数衰减率
optimizer = optim.Adam(net.parameters(),lr=0.0001,betas=(0.9,0.99))
#选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#加载网络
net.to(device)

print("开始训练!")
num_epochs = 50#训练次数
for epoch in range(num_epochs):
    running_loss = 0
    batch_size = 100

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()#梯度初始化为零
        loss.backward()#反向传播
        optimizer.step()#更新所有参数
    print('经过%d个epoch后,损失为:%.4f'%(epoch+1, loss.item()))
print("结束训练")

# #保存训练模型
# torch.save(net, 'vggMNIST12.pkl')
#加载训练模型
net = torch.load('vggMNIST12.pkl')
#开始识别
with torch.no_grad():
    #在接下来的代码中，所有Tensor的requires_grad都会被设置为False
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        out = net(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('测试集图片的准确率是:{}%'.format(100 * correct / total)) #输出识别准确率
