from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as tr
import pandas as pd
from PIL import Image
import skimage.io as io
from skimage import img_as_ubyte
import os.path as osp

class ClassDataset(Dataset):
    def __init__(self, csv_path, data_path, transforms, tg_size):
        # assumes in the csv first column is file name, second column is target
        self.csv_path=csv_path
        df = pd.read_csv(self.csv_path)
        self.data_path = data_path
        self.im_list = df['image_id'].values
        self.targets = (5*df['score'].values).astype(np.uint8)

        self.classes = list(df['score'].unique())
        self.transforms = transforms
        self.resize = tr.Resize(tg_size)
        self.tensorize = tr.ToTensor()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.normalize = tr.Normalize(mean, std)

    def __getitem__(self, index):
        # load image and targets
        img = Image.fromarray(img_as_ubyte(io.imread(osp.join(self.data_path, self.im_list[index])))).convert('RGB')
        target = self.targets[index]

        img = self.resize(img)
        if self.transforms is not None:
            img = self.transforms(img)
        img = self.tensorize(img)
        img = self.normalize(img)
        return img, target

    def __len__(self):
        return len(self.im_list)

def get_class_loaders(csv_train, csv_val, data_path, tg_size, batch_size, num_workers, see_classes=True):
    # First dataset has TrivialAugment transforms, second loader has nothing (resize, tensor, normalize are inside)
    # train_transforms = tr.TrivialAugmentWide()
    # geometric transforms
    h_flip = tr.RandomHorizontalFlip()
    v_flip = tr.RandomVerticalFlip()
    rotate = tr.RandomRotation(degrees=15)

    scale = tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = tr.RandomChoice([scale, transl, rotate])

    # intensity transforms
    brightness, contrast, saturation, hue = 0.10, 0.10, 0.10, 0.01
    jitter = tr.ColorJitter(brightness, contrast, saturation, hue)
    train_transforms = tr.Compose([scale_transl_rot, jitter, h_flip, v_flip])

    val_transforms = None

    train_dataset = ClassDataset(csv_train, data_path, train_transforms, tg_size)
    val_dataset = ClassDataset(csv_val, data_path, val_transforms, tg_size)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    if see_classes:
        print(20 * '*')
        for c in range(len(np.unique(train_dataset.targets))):
            exs_train = np.count_nonzero(train_dataset.targets == c)
            exs_val = np.count_nonzero(val_dataset.targets == c)
            print('Found {:d}/{:d} train/val examples of class {}'.format(exs_train, exs_val, c))

    return train_loader, val_loader

def get_class_test_loader(csv_test, data_path, tg_size, batch_size, num_workers):
    # resize, tensor, normalize are inside ClassDataset already
    test_transforms = None
    test_dataset = ClassDataset(csv_test, data_path, test_transforms, tg_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return test_loader

