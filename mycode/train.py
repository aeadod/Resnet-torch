import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as tranforms
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import net
import utils


data_dir = 'C:/Users/aeadod/Desktop/深度学习/第一次/fashion-mnist/a'
tranform = tranforms.Compose([tranforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, transform=tranform)
val_dataset  = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=tranform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=4, num_workers=0, shuffle=False)

# plt.figure()
# utils.imshow_batch(next(iter(train_dataloader)))
# plt.show()
device = torch.device('cuda:0')
net = net.Net()
print(net)
net = net.to(device)

loss_fc = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# -----------------------------训练-----------------------------------------
file_runing_loss = open('./log/running_loss.txt', 'w')
file_test_accuarcy = open('./log/test_accuracy.txt', 'w')

epoch_num = 100
for epoch in range(epoch_num):
    running_loss = 0.0
    accuracy = 0.0
    scheduler.step()
    for i, sample_batch in enumerate(train_dataloader):

        inputs = sample_batch[0]
        labels = sample_batch[1]

        inputs = inputs.to(device)
        labels = labels.to(device)

        net.train()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fc(outputs, labels)
        loss.backward()
        optimizer.step()

        print(i, loss.item())

        # 统计数据,loss,accuracy
        running_loss += loss.item()
        if i % 20 == 19:
            correct = 0
            total = 0
            net.eval()
            for inputs, labels in val_dataloader:
                outputs = net(inputs)
                _, prediction = torch.max(outputs, 1)

                correct += ((prediction == labels).sum()).item()
                total += labels.size(0)

            accuracy = correct / total
            print('[{},{}] running loss = {:.5f} acc = {:.5f}'.format(epoch + 1, i+1, running_loss / 20, accuracy))
            file_runing_loss.write(str(running_loss / 20)+'\n')
            file_test_accuarcy.write(str(accuracy)+'\n')
            running_loss = 0.0

print('\n train finish')
torch.save(net.state_dict(), './model/model_100_epoch.pth')
