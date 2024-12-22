import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 检查是否可以使用 MPS (Mac), CUDA (NVIDIA) 或 回退到 CPU
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"使用设备: {device}")

# 定义神经网络模型
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 输入通道1，输出通道32，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, 3)  # 输入通道32，输出通道64，卷积核3x3
        self.pool = nn.MaxPool2d(2, 2)  # 2x2的最大池化
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)  # 输出层，10个数字类别

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))  # 第一次卷积+激活
        x = self.pool(nn.functional.relu(self.conv2(x)))  # 第二次卷积+激活+池化
        x = self.dropout1(x)
        x = x.view(-1, 64 * 12 * 12)  # 展平
        x = nn.functional.relu(self.fc1(x))  # 全连接层+激活
        x = self.dropout2(x)
        x = self.fc2(x)  # 输出层
        return nn.functional.log_softmax(x, dim=1)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载训练集和测试集
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 初始化模型、损失函数和优化器
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'损失: {loss.item():.6f}')

# 测试函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\n测试集平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return accuracy

# 主训练循环
def main():
    epochs = 10
    best_accuracy = 0
    
    print("开始训练...")
    for epoch in range(1, epochs + 1):
        train(epoch)
        accuracy = test()
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'mnist_model.pth')
            print(f"模型已保存，当前最佳准确率: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main() 