import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from utils.fashion_mnist import MNIST, FashionMNIST
from utils.get_date import AnimeDataset

def get_data_loader(dataset,batch_size,dataroot,download):

    if dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = MNIST(root=dataroot, train=True, download=download, transform=trans)
        test_dataset = MNIST(root=dataroot, train=False, download=download, transform=trans)

    elif dataset == 'fashion-mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = FashionMNIST(root=dataroot, train=True, download=download, transform=trans)
        test_dataset = FashionMNIST(root=dataroot, train=False, download=download, transform=trans)

    elif dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.CIFAR10(root=dataroot, train=True, download=download, transform=trans)
        test_dataset = dset.CIFAR10(root=dataroot, train=False, download=download, transform=trans)

    elif dataset == 'stl10':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = dset.STL10(root=dataroot, split='train', download=download, transform=trans)
        test_dataset = dset.STL10(root=dataroot,  split='test', download=download, transform=trans)
    elif dataset == 'animedata':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.ImageFolder(root=dataroot, transform=trans)
        test_dataset = dset.ImageFolder(root=dataroot, transform=trans)
    
    # Check if everything is ok with loading datasets
    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader
