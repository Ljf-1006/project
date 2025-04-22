from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import torchvision
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
from tqdm import tqdm
from torchvision.models import ResNet50_Weights

data_dir = r"C:\Ljf\大學7788\專題\dog-breed-identification"
csv_path = os.path.join(data_dir, 'labels.csv')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
assert os.path.exists(csv_path)

train_csv = pd.read_csv(csv_path)
train_csv.head()

count = train_csv['breed'].value_counts()
count
plt.bar(range(120), count)
plt.xlabel("Class")
plt.ylabel("Number of Class")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(train_csv.breed)
print("映射后的标签", labels)
print("训练集的大小", labels.shape)

filenames = train_csv.id.values
filenames_train, filenames_val, labels_train, labels_val = \
    train_test_split(filenames, labels, test_size=0.1, stratify=labels)
print("训练数据数", len(filenames_train))
print("验证数据数", len(filenames_val))

class DogDataset(Dataset):
    """Dog Breed Dataset"""

    def __init__(self, filenames, labels, root_dir, transform=None):
        assert len(filenames) == len(labels)
        self.filenames = filenames
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        label = self.labels[item]
        img_name = os.path.join(self.root_dir, self.filenames[item] + '.jpg')

        with Image.open(img_name) as f:
            img = f.convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.labels is None:
            return img, self.filenames[item]
        else:
            return img, self.labels[item]

train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.75, 1)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])
val_transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                ])
batch_size = 32

train_dataset = DogDataset(filenames_train, labels_train, train_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = DogDataset(filenames_val, labels_val, train_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(len(train_dataset))
print(len(val_dataset))

# from torchvision.models import resnet50, ResNet50_Weights
# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
# from torchvision.models import resnet50
# model = resnet50(weights='DEFAULT')  # 或者 'IMAGENET1K_V1'
# from torchvision.models import resnet50, ResNet50_Weights
# model = resnet50(weights=ResNet50_Weights.DEFAULT)

n_class = 120 # 总共有120种的狗
net = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT) # 得到预训练模型
for param in net.parameters():
    param.requires_grad = False # 固定住所有的权重
net.fc = torch.nn.Linear(2048, n_class) # 将最后的全连接层改掉

criterion = torch.nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = torch.optim.Adam(net.fc.parameters(), lr=0.0001)  # 学习率为0.0001的Adam优化器



# def train_epoch(net, data_iter, criterion, optimizer, use_cuda, print_every=50):
#     net.train()
#     correct = 0.0
#     for batch_idx, (x, y) in tqdm(enumerate(data_iter)):
#         if use_cuda:
#             x, y = x.cuda(), y.cuda()
#         x = Variable(x)
#         y = Variable(y)
#         optimizer.zero_grad()
#         logits = net(x)
#         loss = criterion(logits, y)
#         loss.backward()
#         optimizer.step()
#
#         prediction = torch.argmax(logits, 1)
#         cur_correct = (prediction == y).sum().float()
#         cur_accuracy = cur_correct / x.shape[0]
#         correct += cur_correct
#
#         if batch_idx % print_every == 0:
#             print('current batch: {}/{} ({:.0f}%)\tLoss: {:.6f}\tAcc: {:.6f}'.format(
#                 batch_idx, len(data_iter),
#                 100. * batch_idx / len(data_iter), loss.data.item(), cur_accuracy))
#
#     accuracy = correct / len(data_iter.dataset)
#     print('Train epoch Acc: {:.6f}'.format(accuracy))
#     return accuracy

def train_epoch(net, train_loader, criterion, optimizer, use_cuda):
    net.train()
    running_loss = 0.0
    running_corrects = 0

    for images, labels in train_loader:
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()  # 將圖片和標籤移至 GPU

        labels = labels.long()
        optimizer.zero_grad()  # 清空上一步的梯度
        outputs = net(images)
        loss = criterion(outputs, labels)  # 標籤應為 LongTensor 用來計算交叉熵
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新權重

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    return epoch_loss, epoch_acc.item()

def val_epoch(net, data_iter, criterion, use_cuda):
    test_loss = 0
    correct = 0
    net.eval() # 将网络设置为验证的模式，不会改变参数
    for batch_idx, (x, y) in tqdm(enumerate(data_iter)):
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        y = y.long()
        x = Variable(x)
        y = Variable(y)
        logits = net(x)
        loss = criterion(logits, y)

        test_loss += loss.data.item()
        prediction = torch.argmax(logits, 1)
        cur_correct = (prediction == y).sum().float()
        correct += cur_correct

    test_loss /= len(data_iter.dataset)
    accuracy = correct / len(data_iter.dataset)

    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(data_iter.dataset), 100. * accuracy))

    return accuracy

EPOCHS = 10 # 设置网络迭代最多10次
use_cuda = torch.cuda.is_available() # 查看是否有 GPU 可用
print(use_cuda)

if use_cuda:
    net.cuda() # 将网络搬到 GPU 上

state = {}
state['val_acc'] = []
state['best_val_acc'] = 0
state['lives'] = 4

for epoch in range(EPOCHS):
    print("Epoch: ", epoch+1)
    train_acc = train_epoch(net, train_loader, criterion, optimizer, use_cuda)
    print("Evaluating...")
    val_acc = val_epoch(net, val_loader, criterion, use_cuda)

    state['val_acc'].append(val_acc)
    if val_acc > state['best_val_acc']:
        state['lives'] = 4
        state['best_val_acc'] = val_acc
    else:
        state['lives'] -= 1
        print("Trial left :", state['lives'])
        if state['lives'] == 2:
            optimizer.param_groups[0]['lr'] /= 2
        if state['lives'] == 0:
            break

torch.save(net, "resnet50_dog_breed.pth")
print("Model saved as resnet50_dog_breed.pth")