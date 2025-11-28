import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize, RandomCrop, RandomHorizontalFlip
import time
from tqdm import tqdm


from sgd import SGD
from resnet import ResNet18

import logging
logging.basicConfig(format='%(message)s', filename='./sgd.log', filemode='w', level=logging.ERROR)
def printlin(text):
    print(text)
    logging.error(text)


class SimpleCIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = (x+1.0)/2.0  # scale to [0,1]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_data_loaders(batch_size=128, num_workers=8):
    """
    Return train/test data_loaders for CIFAR-10 with standard augmentation.
    """
    train_dataset = torchvision.datasets.CIFAR10('/work1/lin/jupyter/data', train=True, download=True, transform=Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), Normalize(0.5, 0.5)]))
    test_dataset = torchvision.datasets.CIFAR10('/work1/lin/jupyter/data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def train(model, data_loader, loss_fun, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(data_loader, desc="Training", ncols=100)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = loss_fun(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct/total:.2f}%",
        })


    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, data_loader, loss_fun, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fun(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    num_epochs = 100
    batch_size = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    printlin(f"Using device: {device}")


    model = SimpleCIFAR10CNN().to(device)
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, loss_fun, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fun, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "./cps/best_cifar10_sgd.pth")

        printlin(f"Runtime so far:         {(time.time()-start_time)/60:.2f} min")

        printlin(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}% | "
            f"Best Acc: {best_acc*100:.2f}%"
        )

if __name__ == "__main__":
    main()