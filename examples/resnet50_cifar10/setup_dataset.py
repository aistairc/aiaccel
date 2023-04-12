
from torchvision import datasets


def setup_dataset():
    datasets.CIFAR10(root='./dataset', download=True)


if __name__ == '__main__':
    setup_dataset()
