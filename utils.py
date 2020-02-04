import os,sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from setup_pgd import LinfPGDAttackOT, attack_over_test_data_ot

class MNIST:
    def __init__(self, root):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.MNIST(root=root+'/data', train=True, transform=trans, download=False)
        test_set = datasets.MNIST(root=root+'/data', train=False, transform=trans, download=False)
        
        self.train_data = train_set
        self.test_data = test_set   
        
def loaddata(args):
    if args['dataset'] == 'mnist':
        train_loader = DataLoader(MNIST(args['root']).train_data, batch_size=args['batch_size'], shuffle=args['shuffle'])
        test_loader = DataLoader(MNIST(args['root']).test_data, batch_size=args['batch_size'], shuffle=False)
    elif args['dataset'] == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        trainset = datasets.CIFAR10(root=args['root']+"/data",
                                train=True,download=False,transform=transform_train)        
        train_loader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=args['shuffle'])                
        transform_test = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root=args['root']+"/data",
                                train=False,download=False,transform=transform_test)
        test_loader = DataLoader(testset, batch_size=args['batch_size'], shuffle=False)    
    elif args['dataset'] == 'stl10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = datasets.STL10(root=args['root']+"/data",
                                 split='train',download=False,transform=transform_train)
        testset = datasets.STL10(root=args['root']+"/data",
                                split='test',download=False,transform=transform_test)
        train_loader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=args['shuffle'])
        test_loader = DataLoader(testset, batch_size=args['batch_size'], shuffle=False)
    else:
        print("unknown dataset")
    return train_loader, test_loader

def loadmodel(args):
    if args['dataset'] == 'mnist':
        from setup_ot_mnist import Encoder, Discriminator, SmallCNN
        encoder = Encoder(args['dim_h'],args['n_z'])
        discriminator = Discriminator(args['dim_h'],args['n_z'])
        classifier = SmallCNN(args['n_z'])        
        if args['init'] != None:
            classifier.load_state_dict(torch.load('./models/mnist'+args['init']+'cla'))
            discriminator.load_state_dict(torch.load('./models/mnist'+args['init']+'dis'))
            encoder.load_state_dict(torch.load('./models/mnist'+args['init']+'enc'))
    elif args['dataset'] == 'cifar10':       
        from setup_ot_cifar10 import Encoder, Discriminator, SmallCNN        
        encoder = Encoder(args['n_z'])    
        discriminator = Discriminator(args['dim_h'],args['n_z'])
        classifier = SmallCNN(args['n_z'])
        encoder.feature.load_state_dict(torch.load("./models/cifar10"+args['enc_init']))
        
        if args['init'] != None:
            classifier.load_state_dict(torch.load('./models/cifar10'+args['init']+'cla'))
            encoder.load_state_dict(torch.load('./models/cifar10'+args['init']+'enc'))
            discriminator.load_state_dict(torch.load('./models/cifar10'+args['init']+'dis'))
        
    elif args['dataset'] == 'stl10':
        from setup_ot_stl10 import Encoder, Discriminator, SmallCNN
        encoder = Encoder(args['n_z'],args['dim_h1'])
        discriminator = Discriminator(args['dim_h'],args['n_z'])
        classifier = SmallCNN(args['n_z'])
        
        if args['init'] != None:
            classifier.load_state_dict(torch.load('./models/stl10'+args['init']+'cla'))
            encoder.load_state_dict(torch.load('./models/stl10'+args['init']+'enc'))
            discriminator.load_state_dict(torch.load('./models/stl10'+args['init']+'dis'))
    else:
        print("unknown model")
    return encoder, discriminator, classifier

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def clip_params(module: nn.Module):
    for p in module.parameters():
        p.data.clamp_(-0.01, 0.01)     
        
def testattack(classifier, encoder, test_loader, epsilon, k, a, dataset, use_cuda=True):
    classifier.eval()
    encoder.eval()
    adversary = LinfPGDAttackOT(classifier, encoder, epsilon=epsilon, k=k, a=a, data=dataset)
    param = {
    'test_batch_size': 100,
    'epsilon': epsilon,
    }            
    attack_over_test_data_ot(classifier, encoder, adversary, param, test_loader, use_cuda=use_cuda)
    return

def savefile(file_name, encoder, discriminator, classifier, dataset):
    if file_name != None:
        root = os.path.abspath(os.path.dirname(sys.argv[0]))+"/models/"+dataset
        if not os.path.exists(root):
            os.mkdir(root)
        torch.save(encoder.state_dict(), root+file_name+"enc")
        torch.save(discriminator.state_dict(), root+file_name+"dis")
        torch.save(classifier.state_dict(), root+file_name+"cla")
    return
