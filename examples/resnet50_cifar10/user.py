import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, models, transforms

from aiaccel.util import aiaccel


# Train
def train_func(model, train_loader, optimizer, device, criterion):
    train_correct, train_loss, sum_of_train_data = 0, 0, 0
    model.train()

    for (inputs, labels) in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_loss += loss.item() * inputs.size(0)
        train_correct += (predicted == labels).sum().item()
        sum_of_train_data += inputs.size(0)

    train_loss /= float(sum_of_train_data)
    train_acc = 100. * train_correct / float(sum_of_train_data)

    return train_loss, train_acc


# Validation and Test
def val_test_func(model, val_loader, device, criterion):
    val_correct, val_loss = 0, 0
    model.eval()

    with torch.no_grad():
        for (inputs, labels) in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

    val_loss /= float(len(val_loader.dataset))
    val_acc = 100. * val_correct / float(len(val_loader.dataset))

    return val_loss, val_acc


def main(p):
    # Setup data augmentation
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Setup dataset
    train_val_dataset = datasets.CIFAR10(
        root='./dataset', train=True, transform=train_transform)

    # (train=True) Train dataser 40000 Val dataset 10000
    # Use torch.utils.data.random_split to devide dataset
    train_length = int(len(train_val_dataset) * 0.8)
    val_length = len(train_val_dataset) - train_length
    generator_seed = 0
    train_dataset, _ = torch.utils.data.random_split(
        train_val_dataset, [train_length, val_length],
        generator=torch.Generator().manual_seed(generator_seed))

    train_val_dataset = datasets.CIFAR10(
        root='./dataset', train=True, transform=val_test_transform)
    _, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_length, val_length],
        generator=torch.Generator().manual_seed(generator_seed))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=p["batch_size"], shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=p["batch_size"], shuffle=False, drop_last=False)

    # (train=False) Test dataset 10000
    test_dataset = datasets.CIFAR10(
        root='./dataset', train=False, transform=val_test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=p["batch_size"], shuffle=False)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup model
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    model = nn.DataParallel(model)

    # Setup loss function
    criterion = nn.CrossEntropyLoss()

    # Setup optimizer
    epochs = 100

    optimizer = optim.SGD(model.parameters(), lr=p["lr"],
                          momentum=p["momentum"], weight_decay=p["weight_decay"])
    manager = CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=p["lr"] * p["lr_decay"])

    logger = logging.getLogger(__name__)

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_func(
            model, train_loader, optimizer, device, criterion)

        # Validation
        val_loss, val_acc = val_test_func(model, val_loader, device, criterion)
        print(f"epoch [{epoch}/{epochs}] "
              f"lr = {optimizer.param_groups[0]['lr']:.4f}, "
              f"train acc: {train_acc:.4f} "
              f"train loss: {train_loss:.4f}, val acc: {val_acc:.4f} "
              f"val loss: {val_loss:.4f}")
        logger.info(f"epoch[{epoch}/{epochs}] "
                    f"lr = {optimizer.param_groups[0]['lr']:.4f}, "
                    f"train acc: {train_acc:.4f} "
                    f"train loss: {train_loss:.4f}, val acc: {val_acc:.4f} "
                    f"val loss: {val_loss:.4f}")
        # Update lr
        manager.step()

    # Test
    test_loss, test_acc = val_test_func(model, test_loader, device, criterion)
    print(f"test acc: {test_acc:.4f}, test loss: {test_loss:.4f}")
    logger.info(f"test acc: {test_acc:.4f}, test loss: {test_loss:.4f}")

    # Return val error rate
    return 100. - val_acc


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
