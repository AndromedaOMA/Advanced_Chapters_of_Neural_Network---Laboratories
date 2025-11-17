import sys

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def preprocessing(config):
    if config['dataset']['name'] == 'MNIST':
        train_transformer = transforms.Compose([
            transforms.RandomAffine(degrees=2, translate=[0.1, 0.1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        train_dataset = datasets.MNIST(root=f'{config["dataset"]["data_dir"]}/train', train=True, transform=train_transformer, download=True)
        test_dataset = datasets.MNIST(root=f'{config["dataset"]["data_dir"]}/test', train=False, transform=test_transformer, download=True)
        train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, pin_memory=True)
        return train_loader, test_loader

    elif config['dataset']['name'] == 'CIFAR10':
        mean = list(map(float, config["dataset"]["mean"]))
        std = list(map(float, config["dataset"]["std"]))

        train_transformer = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = datasets.CIFAR10(root=f'{config["dataset"]["data_dir"]}/train', train=True, transform=train_transformer, download=True)
        test_dataset = datasets.CIFAR10(root=f'{config["dataset"]["data_dir"]}/test', train=False, transform=test_transformer, download=True)
        train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, pin_memory=True)
        return train_loader, test_loader

    elif config['dataset']['name'] == 'CIFAR100':
        mean = list(map(float, config["dataset"]["mean"]))
        std = list(map(float, config["dataset"]["std"]))

        train_transformer = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = datasets.CIFAR100(root=f'{config["dataset"]["data_dir"]}/train', train=True, transform=train_transformer, download=True)
        test_dataset = datasets.CIFAR100(root=f'{config["dataset"]["data_dir"]}/test', train=False, transform=test_transformer, download=True)
        train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, pin_memory=True)
        return train_loader, test_loader

    elif config['dataset']['name'] == 'OxfordIIITPet':
        mean = list(map(float, config["dataset"]["mean"]))
        std = list(map(float, config["dataset"]["std"]))

        train_transformer = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = datasets.OxfordIIITPet(root=f'{config["dataset"]["data_dir"]}/train', train=True, transform=train_transformer, download=True)
        test_dataset = datasets.OxfordIIITPet(root=f'{config["dataset"]["data_dir"]}/test', train=False, transform=test_transformer, download=True)
        train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, pin_memory=True)
        return train_loader, test_loader
    else:
        print('The dataset name you have entered is not supported!')
        sys.exit()

