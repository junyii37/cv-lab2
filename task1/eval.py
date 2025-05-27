import os
import numpy as np
import random
from tqdm import tqdm
import argparse
import json
import pickle

import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset


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
parser.add_argument('--model_path', type=str, default='best_model.pth')
parser.add_argument('--data_dir', type=str, default='../caltech-101/caltech-101/101_ObjectCategories')
parser.add_argument('--idx_path', type=str, default='dataset_idx.pkl')
parser.add_argument('--output_dir', type=str, default='results')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)

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

# split test dataset
with open(args.idx_path, 'rb') as f:
    dataset_idx = pickle.load(f)
test_idx = dataset_idx['test']
test_dataset = Subset(data, test_idx)

print('Datasets initialized!')

# init dataloaders
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

print('Dataloaders initialized!')

# init model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
model.load_state_dict(torch.load(args.model_path))
model = model.to(device)
print('Model initialized!')

# eval
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        total += len(labels)
        correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    print(f'[Test]Accuracy:{acc:.4f}')

output_path = os.path.join(args.output_dir, args.model_path.split('.')[0] + '.txt')
os.makedirs(args.output_dir, exist_ok=True)
with open(output_path, 'w') as f:
    f.write(f'[Test]Accuracy:{acc:.4f}')







