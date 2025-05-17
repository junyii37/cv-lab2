import numpy as np

from tqdm import tqdm

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dataset
data_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = '../caltech-101/caltech-101/101_ObjectCategories'
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
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True)
                for x in ['train', 'valid', 'test']}
print('Dataloaders initialized!')

# init model
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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
    {'params': fc_params, 'lr': 0.01},
    {'params': other_params, 'lr': 0.0001}
], momentum=0.9, weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

criterion = nn.CrossEntropyLoss()
print('Optimizer initialized!')

# train
best_acc = 0.0
log_iter = 50
patience = 3
print('Begin Training...')
for epoch in range(30):
    print(f'[Train]Epoch {epoch+1}')
    model.train()
    # train
    for inputs, labels in tqdm(dataloaders['train']):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # if not iter % log_iter:
        #     print(f'[Train]Iter:{iter}, Loss:{loss.item():.4f}')

    # valid
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += len(labels)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total
    print(f'[Valid]Accuracy:{val_acc:.4f}')

    # save models and early stop
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience -= 1
        if patience <= 0:
            print(f'Early stop at epoch {epoch+1}')
            break

    # schedule learning rate
    scheduler.step()

print('Training completed!')

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







