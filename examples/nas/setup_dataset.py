from torchvision import datasets


def setup_dataset(cifar10=False, cifar100=False):
    if cifar10:
        datasets.CIFAR10(root="./dataset_cifar10", download=True)
    elif cifar100:
        datasets.CIFAR100(root="./dataset_cifar100", download=True)


if __name__ == "__main__":
    setup_dataset(cifar10=True)
