import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# 定义和训练时同样的模型结构
class MultiTaskModel(nn.Module):
    def __init__(self, num_kinds, num_days):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)  # 预测时不用加载预训练权重
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.kind_classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_kinds)
        )
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

# 请替换成你训练时的类别列表
kind_classes = ['FZX', 'NMZ', 'GW']  # 例如 ['lychee', 'mango', 'banana']
days_classes = ['1', '2','3','4','5','6']  # 例如采后天数标签

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
num_kinds = len(kind_classes)
num_days = len(days_classes)
model = MultiTaskModel(num_kinds=num_kinds, num_days=num_days)
model.load_state_dict(torch.load(r"D:\Vscode\python_code\IGEM\best_model.pth", map_location=device))
model.to(device)
model.eval()

# 图像预处理，和训练时保持一致
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)  # 增加batch维度，并转GPU/CPU

    with torch.no_grad():
        kind_logits, days_logits = model(img_tensor)

        kind_pred_idx = torch.argmax(kind_logits, dim=1).item()
        days_pred_idx = torch.argmax(days_logits, dim=1).item()

    kind_pred = kind_classes[kind_pred_idx]
    days_pred = days_classes[days_pred_idx]

    return kind_pred, days_pred

if __name__ == "__main__":
    test_img_path = r"C:\Users\16270\Documents\WeChat Files\wxid_g9jxbli2tq9u22\FileStorage\File\2025-06\lychee_dataset\images\rgb\sample_140_rgb.jpg"  # 替换成你想预测的图片路径
    kind_pred, days_pred = predict(test_img_path)
    print(f"预测类别: {kind_pred}, 采后天数: {days_pred}")