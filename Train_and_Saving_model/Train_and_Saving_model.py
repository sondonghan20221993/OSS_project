import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ---------------------
# 1. 데이터셋 & 전처리
# ---------------------
data_dir = "/kaggle/input/deepfake-database/deepfake_database"
batch_size = 32

transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "validation": transforms.Compose([   # 🔑 validation 추가
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([         # 🔑 test 추가
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# 📌 폴더 경로 각각 지정
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform["train"])
val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "validation"), transform["validation"])
test_dataset  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform["test"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

num_classes = len(train_dataset.classes)  # 라벨 개수 자동 추출

# ---------------------
# 2. 모델 정의
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# ---------------------
# 3. Loss & Optimizer
# ---------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# ---------------------
# 4. 학습 루프
# ---------------------
best_acc = 0.0
num_epochs = 10
save_path = "best_model.pth"

#지능형 리스트로 미리 할당
train_loss_list = [None for _ in range(num_epochs)]
train_acc_list = [None for _ in range(num_epochs)]
val_loss_list = [None for _ in range(num_epochs)]
val_acc_list = [None for _ in range(num_epochs)] 

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

    # ---- Training ----
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ---- Validation ----
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    val_loss /= total
    val_acc = correct / total

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    #해야할것 (결과시각화를 위한 값 저장)----------------------------------------
    """
    해당 에포크(반복횟수에)
    train_loss, val  #학습 손실, 정확도
    val_loss, val #평가 손실, 정확도
    이 임시변수로 저장되어있다.
    이것을 train_loss_list ..으로 만들어놓은 저장공간에 넣어보자
    ex)
    train_loss_list[num_epochs] = train loss #이렇게하면 해당 반복횟수의 값을 리스트에 넣는다.
    
    """
    # ---- Save Best Model ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"✅ 모델 저장됨: {save_path}")
#해야할것 (결과시각화)----------------------------------------
"""
matplotlib함수를 이용해서 
train_loss_lsit, train_acc_list, val_loss_lsit, val_acc_lsit

fig = plt.figure()
plt.plot(list([n for n in range(1, num_epochs+1)]), list(value for value in train_loss_list),marker='o', linestyle='-', label="train_loss") 
plt.legend()
plt.show()
#n for n in range(1, num_epochs+1)로 반복횟쉬 x축으로 지정
#train_loss_list값을 반복문으로 꺼냄

이러한 형식으로 train_acc_list, val모두 plot을 이용해서 그리자


"""
print(f"\n🎯 학습 완료! 최고 정확도: {best_acc:.4f}")
