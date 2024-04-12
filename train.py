# %%
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import AutoEncoder
from tqdm import tqdm 
import numpy as np

# %%
save_path = 'model.pth'

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
    transforms.Resize((256, 256)),  # 调整图片大小
    transforms.RandomRotation(10),
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #[-1, 1]
])

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图片大小
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #[-1, 1]
])

# 数据集路径
base_path = '/Group16T/raw_data/covid_cxr'

# 加载数据集
train_dataset = datasets.ImageFolder(root=f'{base_path}/train', transform=transform_train)
val_dataset = datasets.ImageFolder(root=f'{base_path}/val', transform=transform)
test_dataset = datasets.ImageFolder(root=f'{base_path}/test', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=16)

print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}, test_dataset: {len(test_dataset)}")

# 初始化模型
autoencoder = AutoEncoder().to(device)

# 损失函数和优化器
loss_func = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)


# %%
num_epochs = 5

autoencoder.train() #Set model to training mode
for epoch in range(num_epochs): 
    epoch_start_time = datetime.now()  # 记录epoch开始的时间 

    train_loss = 0
    # 使用tqdm包装数据加载器，以显示进度条
    train_loader_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    for data in train_loader_pbar:
        img, _ = data  # img is in shape (batch_size, 3, 64, 64) / I don't need label and patient_id in training
        img_flat = img.view(img.size(0), -1)  # Flatten img for the model input
        img_flat = Variable(img_flat).to(device)
        # ===================forward=====================
        output = autoencoder(img_flat)
        output = output.view(img.size(0), 3, 256, 256)  # Reshape output to original shape for loss calculation
        loss = loss_func(output, img.to(device))  # Use original shape img for loss calculation
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================       
    train_loader_pbar.set_description(
                f'Epoch {epoch+1}/{num_epochs} '
                f'[Train] Loss: {train_loss/len(train_loader_pbar):.4f}')
    

#Save model
torch.save(autoencoder.state_dict(), save_path)

   

# %%

autoencoder.eval()
val_results = {
    "positive": [],
    "negative": []
}

loss_func = nn.MSELoss(reduction='none')
# 使用tqdm包装数据加载器，以显示进度条
val_loader_pbar = tqdm(val_loader, desc=f'[Val] Single Epoch Validation')
with torch.no_grad():
    for data in val_loader_pbar:
        img, label = data  # img is in shape (batch_size, 3, 64, 64)
        img_flat = img.view(img.size(0), -1)  # Flatten img for the model input
        img_flat = Variable(img_flat).to(device)
        # ===================forward=====================
        output = autoencoder(img_flat)
        output = output.view(img.size(0), 3, 256, 256)  # Reshape output to original shape for loss calculation
        loss = loss_func(output, img.to(device))  # Use original shape img for loss calculation
        loss_per_sample = loss.mean(dim=[1, 2, 3])  # 平均每個樣本的損失，假設是圖像數據

        for i in range(len(label)):
            sample_loss = loss_per_sample[i].item()
            if label[i] == 0:
                val_results["negative"].append(sample_loss)
            else:
                val_results["positive"].append(sample_loss)
                
        val_loader_pbar.set_description(
                    f'Epoch {epoch+1}/{num_epochs} '
                    f'[Val] Loss: {loss.mean().item()/len(val_loader_pbar):.4f}')


# %%
print(val_results)
negative_data = np.array(val_results["negative"])
print(negative_data)

# %%
#Calculate 95 precentile of negative data
threshold = np.percentile(negative_data, 50)
print(f"Threshold: {threshold}")

def thresholding(x, threshold):
    if x > threshold:
        return 1
    else:
        return 0
    

# %%
# 初始化列表以存储真实标签和模型预测概率
y_true = []
test_loss = []
y_pred = []


autoencoder.eval()
# 使用tqdm包装数据加载器，以显示进度条
test_loader_pbar = tqdm(test_loader, desc=f'[Test]')
with torch.no_grad():
    for data in test_loader_pbar:
        img, label = data  # img is in shape (batch_size, 3, 64, 64)
        img_flat = img.view(img.size(0), -1)  # Flatten img for the model input
        img_flat = Variable(img_flat).to(device)
        # ===================forward=====================
        output = autoencoder(img_flat)
        output = output.view(img.size(0), 3, 256, 256)  # Reshape output to original shape for loss calculation
        loss = loss_func(output, img.to(device))  # Use original shape img for loss calculation
        loss_per_sample = loss.mean(dim=[1, 2, 3])  # 平均每個樣本的損失，假設是圖像數據
        
        # Save the results
        for i in range(len(label)):
            sample_loss = loss_per_sample[i].item()
            y_true.append(label[i].item())
            test_loss.append(sample_loss)
                
        val_loader_pbar.set_description(
                    f'Epoch {epoch+1}/{num_epochs} '
                    f'[Test] Loss: {loss.mean().item()/len(test_loader_pbar):.4f}')

# %%
roc_data = []

#Calculate ROC

for threshold in np.linspace(0, 1, 100):
    y_pred = [thresholding(x, threshold) for x in test_loss]
    tp = sum([1 for i in range(len(y_pred)) if y_pred[i] == 1 and y_true[i] == 1])
    tn = sum([1 for i in range(len(y_pred)) if y_pred[i] == 0 and y_true[i] == 0])
    fp = sum([1 for i in range(len(y_pred)) if y_pred[i] == 1 and y_true[i] == 0])
    fn = sum([1 for i in range(len(y_pred)) if y_pred[i] == 0 and y_true[i] == 1])
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    roc_data.append((tpr, fpr))

# %%
#Draw ROC curve
import matplotlib.pyplot as plt

roc_data = sorted(roc_data, key=lambda x: x[1])
roc_data = [(0, 0)] + roc_data + [(1, 1)]

plt.plot([x[1] for x in roc_data], [x[0] for x in roc_data])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# %%
print(y_true)
print(y_pred)

# %%
# 将列表转换为numpy数组以便进行向量化计算
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 计算TP, TN, FP, FN
TP = np.sum((y_pred == 1) & (y_true == 1))
TN = np.sum((y_pred == 0) & (y_true == 0))
FP = np.sum((y_pred == 1) & (y_true == 0))
FN = np.sum((y_pred == 0) & (y_true == 1))


# 基于TP, TN, FP, FN计算准确率、精确率、召回率（敏感性）、特异性
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")

# %%



