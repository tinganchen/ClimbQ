#from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

class syndigits_LT(Dataset):
    def __init__(self, args, root, train, transform):
        self.args = args
        self.root = root
        if train:
            data_csv = self.root + f'syndigits_train{args.lt_gamma}.csv'
        else:
            data_csv = self.root + 'syndigits_test.csv'
        
        self.data = pd.read_csv(data_csv)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data.iloc[i]['image'], self.data.iloc[i]['label']
        path = os.path.join(self.args.data_dir, path)
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
        
        
        
class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = False

        transform_train = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        trainset = syndigits_LT(args=args, root=args.csv_dir, train=True, transform=transform_train)
        
        self.loader_train = DataLoader(
            trainset, batch_size=args.train_batch_size, shuffle=True, 
            num_workers=2, pin_memory=pin_memory
            )

        testset = syndigits_LT(args=args, root=args.csv_dir, train=False, transform=transform_test)
        
        self.loader_test = DataLoader(
            testset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=2, pin_memory=pin_memory)
