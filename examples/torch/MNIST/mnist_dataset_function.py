import torchvision
from torchvision import transforms


def train_dataset():
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    return torchvision.datasets.MNIST("./dataset", train=True, download=True, transform=transform)


def val_dataset():
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    return torchvision.datasets.MNIST("./dataset", train=False, download=True, transform=transform)
