import os
import numpy as np
import time
import random
from tqdm import tqdm
import argparse
import json

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get arguments from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/caltech-101/caltech-101/101_ObjectCategories')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--finetune_lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)

parser.add_argument('--step_size', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.1)

parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--log_iter', type=int, default=50, help='log training loss every n iterations')
parser.add_argument('--unpretrained', action='store_true', help='forbid  pretrained model')

args = parser.parse_args()

# load dataset
data_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = args.data_dir
data = datasets.ImageFolder(root=data_dir, transform=data_transforms)
num_classes = len(data.classes)

# split dataset(train: valid: test = 8:1:1)
targets = np.array(data.targets)
train_idx, valid_and_test_idx = train_test_split(
    np.arange(len(data)),
    test_size=0.2,
    random_state=42,
    stratify=targets
)

train_data = Subset(data, train_idx)

valid_and_test_targets = targets[valid_and_test_idx]

valid_idx, test_idx = train_test_split(
    valid_and_test_idx,
    test_size=0.5,
    random_state=42,
    stratify=valid_and_test_targets
)

valid_data = Subset(data, valid_idx)
test_data = Subset(data, test_idx)

image_datasets = {
    'train': train_data,
    'valid': valid_data,
    'test': test_data
}
print('Datasets initialized!')

# init dataloaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                for x in ['train', 'valid', 'test']}
print('Dataloaders initialized!')

# init model
weights = ResNet18_Weights.IMAGENET1K_V1 if args.unpretrained is False else None
print("weights:", weights)
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
model = model.to(device)
print('Model initialized!')

# init optimizer and loss function
fc_params = []
other_params = []
for name, param in model.named_parameters():
    if "fc" in name:
        fc_params.append(param)
    else:
        other_params.append(param)

optimizer = optim.SGD([
    {'params': fc_params, 'lr': args.lr},
    {'params': other_params, 'lr': args.finetune_lr}
], momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

criterion = nn.CrossEntropyLoss()
print('Optimizer initialized!')

# train
log_dir = f'logs/{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
cfg = {
    'data_dir': args.data_dir,
    'batch__size': args.batch_size,
    'num_workers': args.num_workers,
    'lr': args.lr,
    'finetune_lr': args.finetune_lr,
    'momentum': args.momentum,
    'weight_decay': args.weight_decay,
    'step_size': args.step_size,
    'gamma': args.gamma,
    'epochs': args.epochs,
    'log_iter': args.log_iter,
    'unpretrained': args.unpretrained
}
print(f"[Training Config]{cfg}")

os.makedirs(log_dir, exist_ok=True)
cfg_path = f'{log_dir}/cfg.json'
with open(cfg_path, 'w') as f:
    json.dump(cfg, f)

writer = SummaryWriter(log_dir=log_dir)
best_acc = 0.0
log_iter = args.log_iter
patience = 3
print('Begin Training...')
for epoch in range(args.epochs):
    print(f'[Train]Epoch {epoch+1}')
    model.train()
    # train
    total_loss = 0.0
    for iter, (inputs, labels) in enumerate(dataloaders['train']):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if not iter % log_iter:
            writer.add_scalar('Train/batch_loss', loss.item(), epoch * len(dataloaders['train']) + iter)
    
    writer.add_scalar('Train/avg_loss', total_loss / len(dataloaders['train']), epoch)

    # valid
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for iter, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += len(labels)
            correct += (preds == labels).sum().item()

    writer.add_scalar('Valid/avg_loss', total_loss / len(dataloaders['valid']), epoch)
    val_acc = 100 * correct / total
    print(f'[Valid]Accuracy:{val_acc:.4f}')
    writer.add_scalar('Valid/acc', val_acc, epoch)

    # save models and early stop
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'models/best_model.pth')
    else:
        patience -= 1
        if patience <= 0:
            print(f'Early stop at epoch {epoch+1}')
            break

    # schedule learning rate
    scheduler.step()

print('Training completed!')
writer.close()

# eval
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        total += len(labels)
        correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    print(f'[Test]Accuracy:{acc:.4f}')

result_dir = f'{log_dir}/result.txt'
with open(result_dir, 'w') as f:
    f.write(f'[Test]Accuracy:{acc:.4f}')







