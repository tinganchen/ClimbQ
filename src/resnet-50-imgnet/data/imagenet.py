import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class Data:
    def __init__(self, args):
        pin_memory = False
        if args.gpus is not None:
            pin_memory = True

        scale_size = 299 if args.target_model.startswith('inception') else 224

        traindir = os.path.join(args.train_data_dir, 'train')
        valdir = os.path.join(args.test_data_dir, 'test')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(scale_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ]))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.loader_train = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True, 
            num_workers=2, pin_memory=pin_memory
            )

        self.loader_test = DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(scale_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.eval_batch_size, shuffle=False,
        num_workers=2, pin_memory=True)