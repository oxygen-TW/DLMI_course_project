# %%
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Subset
from model import AutoEncoder, ConvAEN
from tqdm import tqdm 
import numpy as np

# %%
save_path = 'model_Conv.pth'
num_epochs = 25

# %%
if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Exiting...")
    exit()

# %%
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform_train = transforms.Compose([
    # transforms.Resize((128, 128)),  # 调整图片大小
    transforms.CenterCrop((128, 128)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #[-1, 1]
])

transform = transforms.Compose([
    # transforms.Resize((128, 128)),  # 调整图片大小
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #[-1, 1]
])

# 数据集路径
base_path = '/Group16T/raw_data/covid_cxr'

# 加载数据集
train_dataset = datasets.ImageFolder(root=f'{base_path}/train', transform=transform_train)

# 過濾出特定類別的索引
neg_class_indices = [i for i, (_, label) in enumerate(train_dataset.samples) if train_dataset.classes[label] == 'negative']

# 使用這些索引來建立一個子集
negative_train_dataset = Subset(train_dataset, neg_class_indices)

# 创建数据加载器
train_loader = DataLoader(negative_train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=16)

print(f"train_dataset: {len(train_dataset)}, negative_train_dataset: {len(negative_train_dataset)}")

# 初始化模型
# autoencoder = AutoEncoder().to(device)
autoencoder = ConvAEN().to(device)

print(autoencoder)

# 损失函数和优化器
loss_func = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)


# %%
autoencoder.train() #Set model to training mode
train_loss = []
for epoch in range(num_epochs): 
    epoch_start_time = datetime.now()  # 记录epoch开始的时间 

    # 使用tqdm包装数据加载器，以显示进度条
    loss = 0
    train_loader_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    for data in train_loader_pbar:
        img, _ = data  # img is in shape (batch_size, 3, 64, 64) / I don't need label and patient_id in training
        # img_flat = img.view(img.size(0), -1)  # Flatten img for the model input
        # img_flat = Variable(img_flat).to(device)
        img = img.to(device)
        # ===================forward=====================
        output = autoencoder.forward(img)
        output = output.view(img.size(0), 3, 128, 128)  # Reshape output to original shape for loss calculation
        loss = loss_func(output, img.to(device))  # Use original shape img for loss calculation
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss.append(loss.item())
    # ===================log========================       
    # train_loader_pbar.set_postfix(f'Loss: {train_loss/len(train_loader_pbar):.4f}')
    

#Save model
torch.save(autoencoder.state_dict(), save_path)
print(f"Model saved to {save_path}")
# %%
