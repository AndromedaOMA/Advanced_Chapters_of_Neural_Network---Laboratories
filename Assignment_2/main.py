# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T14:47:09.913094Z","iopub.execute_input":"2025-10-10T14:47:09.913353Z","iopub.status.idle":"2025-10-10T14:47:19.56024Z","shell.execute_reply.started":"2025-10-10T14:47:09.913329Z","shell.execute_reply":"2025-10-10T14:47:19.559394Z"}}
import os

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import v2
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
from tqdm import tqdm
import numpy as np
import pickle

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T14:47:19.561078Z","iopub.execute_input":"2025-10-10T14:47:19.561478Z","iopub.status.idle":"2025-10-10T14:47:19.65194Z","shell.execute_reply.started":"2025-10-10T14:47:19.561458Z","shell.execute_reply":"2025-10-10T14:47:19.65113Z"}}
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
enable_half = device.type != "cpu"
scaler = GradScaler(device, enabled=enable_half)

print("Grad scaler is enabled:", enable_half)
device

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T14:47:19.653523Z","iopub.execute_input":"2025-10-10T14:47:19.653769Z","iopub.status.idle":"2025-10-10T14:47:19.664652Z","shell.execute_reply.started":"2025-10-10T14:47:19.653751Z","shell.execute_reply":"2025-10-10T14:47:19.664087Z"}}
if os.path.exists("/kaggle/input") and os.path.exists("/kaggle/working"):
    print("Running on Kaggle.")
    SVHN_test = "/kaggle/input/fii-atnn-2025-competition-2/SVHN_test.pkl"
    SVHN_train = "/kaggle/input/fii-atnn-2025-competition-2/SVHN_train.pkl"
else:
    print("Not on Kaggle.")
    SVHN_test = "data/SVHN_test.pkl"
    SVHN_train = "data/SVHN_train.pkl"


# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T14:47:19.665405Z","iopub.execute_input":"2025-10-10T14:47:19.665641Z","iopub.status.idle":"2025-10-10T14:47:19.680769Z","shell.execute_reply.started":"2025-10-10T14:47:19.665623Z","shell.execute_reply":"2025-10-10T14:47:19.680177Z"}}
class SVHN_Dataset(Dataset):
    def __init__(self, train: bool, transforms: v2.Transform):
        path = SVHN_test
        if train:
            path = SVHN_train
        with open(path, "rb") as fd:
            self.data = pickle.load(fd)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        image, label = self.data[i]
        if self.transforms is None:
            return image, label
        return self.transforms(image), label


cudnn.benchmark = True

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T14:47:19.681631Z","iopub.execute_input":"2025-10-10T14:47:19.681912Z","iopub.status.idle":"2025-10-10T14:47:24.8572Z","shell.execute_reply.started":"2025-10-10T14:47:19.681875Z","shell.execute_reply":"2025-10-10T14:47:24.856515Z"}}
basic_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])
# train_transforms = v2.Compose([
#     v2.ToImage(),
#     v2.RandomCrop(32, padding=4),
#     # v2.RandomHorizontalFlip(),
#     # v2.ColorJitter(0.2, 0.2, 0.2),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
# ])

# test_transforms = v2.Compose([
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
# ])

train_set = SVHN_Dataset(train=True, transforms=basic_transforms)
test_set = SVHN_Dataset(train=False, transforms=basic_transforms)

# train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=500)


# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T14:47:24.85801Z","iopub.execute_input":"2025-10-10T14:47:24.858272Z","iopub.status.idle":"2025-10-10T14:47:24.86752Z","shell.execute_reply.started":"2025-10-10T14:47:24.858248Z","shell.execute_reply":"2025-10-10T14:47:24.866811Z"}}
class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()

        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Classifier
            nn.Flatten(),
            nn.Linear(512, 100)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T14:47:24.868266Z","iopub.execute_input":"2025-10-10T14:47:24.868446Z","iopub.status.idle":"2025-10-10T14:47:25.305745Z","shell.execute_reply.started":"2025-10-10T14:47:24.868431Z","shell.execute_reply":"2025-10-10T14:47:25.304974Z"}}
model = VGG13().to(device)
model = torch.jit.script(model)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, fused=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
# https://www.kaggle.com/code/nnisarg/svnh-alexnet-95-accuracy
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)


# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T14:47:25.306501Z","iopub.execute_input":"2025-10-10T14:47:25.306761Z","iopub.status.idle":"2025-10-10T14:47:25.312783Z","shell.execute_reply.started":"2025-10-10T14:47:25.306733Z","shell.execute_reply":"2025-10-10T14:47:25.312077Z"}}
def train():
    model.train()
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T14:47:25.314655Z","iopub.execute_input":"2025-10-10T14:47:25.314887Z","iopub.status.idle":"2025-10-10T14:47:25.328055Z","shell.execute_reply.started":"2025-10-10T14:47:25.31487Z","shell.execute_reply":"2025-10-10T14:47:25.327343Z"}}
@torch.inference_mode()
def inference():
    model.eval()

    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels


# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T14:47:25.328832Z","iopub.execute_input":"2025-10-10T14:47:25.329026Z","iopub.status.idle":"2025-10-10T14:48:04.559905Z","shell.execute_reply.started":"2025-10-10T14:47:25.329012Z","shell.execute_reply":"2025-10-10T14:48:04.559333Z"}}
best = 0.0
best_epoch = 0
epochs = list(range(50))

with tqdm(epochs) as tbar:
    for epoch in tbar:
        train_acc = train()
        scheduler.step()

        if train_acc > best:
            best = train_acc
            best_epoch = epoch

        tbar.set_description(f"Train: {train_acc:.2f}, Best: {best:.2f} at epoch {best_epoch}")

# %% [code] {"execution":{"iopub.status.busy":"2025-10-10T14:48:04.56058Z","iopub.execute_input":"2025-10-10T14:48:04.560819Z","iopub.status.idle":"2025-10-10T14:48:06.734492Z","shell.execute_reply.started":"2025-10-10T14:48:04.560794Z","shell.execute_reply":"2025-10-10T14:48:06.733929Z"}}
data = {
    "ID": [],
    "target": []
}

for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
df.to_csv("/kaggle/working/submission.csv", index=False)