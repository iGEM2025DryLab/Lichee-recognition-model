import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class HarvestDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # 对kind和days_after_harvest进行标签编码
        self.le_kind = LabelEncoder()
        self.df['kind_label'] = self.le_kind.fit_transform(self.df['kind'])
        self.le_days = LabelEncoder()
        self.df['days_label'] = self.le_days.fit_transform(self.df['days_after_harvest'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['rgb_image'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        kind_label = torch.tensor(row['kind_label']).long()
        days_label = torch.tensor(row['days_label']).long()

        return image, kind_label, days_label

class MultiTaskModel(nn.Module):
    def __init__(self, num_kinds, num_days):
        super().__init__()
        # 使用预训练ResNet18做特征提取
        self.backbone = models.resnet34(pretrained=True)
        # 替换最后分类层，移除fc层
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 分类头
        self.kind_classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_kinds)
        )
        # 天数预测头
        self.days_classifier = nn.Sequential(
            nn.Linear(num_ftrs, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_days)
        )

    def forward(self, x):
        features = self.backbone(x)
        kind_out = self.kind_classifier(features)
        days_out = self.days_classifier(features)
        return kind_out, days_out

def train(csv_path, img_dir, batch_size=16, epochs=20, lr=1e-3):

    best_acc_sum = 0.0
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = HarvestDataset(csv_path, img_dir, transform)
    # 划分训练验证集
    train_df, val_df = train_test_split(dataset.df, test_size=0.2, stratify=dataset.df['kind'], random_state=42)

    train_dataset = HarvestDataset(csv_path, img_dir, transform)
    train_dataset.df = train_df.reset_index(drop=True)
    train_dataset.le_kind = dataset.le_kind
    train_dataset.le_days = dataset.le_days

    val_dataset = HarvestDataset(csv_path, img_dir, transform)
    val_dataset.df = val_df.reset_index(drop=True)
    val_dataset.le_kind = dataset.le_kind
    val_dataset.le_days = dataset.le_days

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskModel(num_kinds=len(dataset.le_kind.classes_), num_days=len(dataset.le_days.classes_)).to(device)

    criterion_kind = nn.CrossEntropyLoss()
    criterion_days = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_kind_correct = 0
        running_days_correct = 0
        total = 0

        for imgs, kind_labels, days_labels in train_loader:
            imgs = imgs.to(device)
            kind_labels = kind_labels.to(device)
            days_labels = days_labels.to(device)

            optimizer.zero_grad()
            out_kind, out_days = model(imgs)

            loss_kind = criterion_kind(out_kind, kind_labels)
            loss_days = criterion_days(out_days, days_labels)
            loss = loss_kind + loss_days

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, kind_preds = torch.max(out_kind, 1)
            _, days_preds = torch.max(out_days, 1)

            running_kind_correct += (kind_preds == kind_labels).sum().item()
            running_days_correct += (days_preds == days_labels).sum().item()
            total += imgs.size(0)

        epoch_loss = running_loss / total
        kind_acc = running_kind_correct / total
        days_acc = running_days_correct / total

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} Kind Acc: {kind_acc:.4f} Days Acc: {days_acc:.4f}")

        # 验证阶段
        model.eval()
        val_kind_correct = 0
        val_days_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, kind_labels, days_labels in val_loader:
                imgs = imgs.to(device)
                kind_labels = kind_labels.to(device)
                days_labels = days_labels.to(device)
                out_kind, out_days = model(imgs)
                _, kind_preds = torch.max(out_kind, 1)
                _, days_preds = torch.max(out_days, 1)
                val_kind_correct += (kind_preds == kind_labels).sum().item()
                val_days_correct += (days_preds == days_labels).sum().item()
                val_total += imgs.size(0)

        val_kind_acc = val_kind_correct / val_total
        val_days_acc = val_days_correct / val_total
        val_acc_sum = val_kind_acc + val_days_acc

        print(f"Validation - Kind Acc: {val_kind_acc:.4f} Days Acc: {val_days_acc:.4f}")

        # 保存表现最好的模型
        if val_acc_sum > best_acc_sum:
            best_acc_sum = val_acc_sum
            save_path = "best_model_resnet_34.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1}: 保存了新的最佳模型，Acc之和为 {val_acc_sum:.4f}")

    print("训练结束，最佳模型已保存。")

if __name__ == "__main__":
    # 请修改这里路径
    CSV_PATH = r"C:\Users\16270\Documents\WeChat Files\wxid_g9jxbli2tq9u22\FileStorage\File\2025-06\lychee_dataset\metadata.csv"
    IMG_DIR = r"C:\Users\16270\Documents\WeChat Files\wxid_g9jxbli2tq9u22\FileStorage\File\2025-06\lychee_dataset\images\rgb"  # 这里放所有rgb图像文件

    train(CSV_PATH, IMG_DIR, batch_size=16, epochs=20, lr=1e-3)