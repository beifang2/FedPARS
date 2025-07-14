import torchvision.transforms as tfs
import os
from torch.utils import data
from osgeo import gdal

class Dataset(data.Dataset):
    def __init__(self,path_root="../data/", mode="train", client_name="austin"):
        super(Dataset,self).__init__()
        self.path_root = os.path.join(path_root + mode,client_name)
        self.rs_images_dir = os.listdir(os.path.join(self.path_root, "image"))
        self.rs_images = [os.path.join(self.path_root, "image", img) for img in self.rs_images_dir]
        self.gt_images_dir = os.listdir(os.path.join(self.path_root,"label"))
        self.gt_images = [os.path.join(self.path_root,"label",img) for img in self.rs_images_dir]

    def __getitem__(self, item):
        img = gdal.Open(self.rs_images[item])
        label = gdal.Open(self.gt_images[item])
        img = img.ReadAsArray().transpose(1, 2, 0)
        label = label.ReadAsArray()
        img = img / 255.0
        label = label / 255.0
        img = tfs.ToTensor()(img)
        label = tfs.ToTensor()(label)

        return img, label

    def __len__(self):
        return len(self.rs_images)

class Dataset_Central(data.Dataset):
    def __init__(self, path_root="../data/", mode="train"):
        super(data.Dataset, self).__init__()
        subfolders = [name for name in os.listdir(path_root + mode)
                      if os.path.isdir(os.path.join(path_root + mode, name))]
        self.rs_images = []
        self.gt_images = []
        for client_name in subfolders:
            self.path_root = os.path.join(path_root + mode, client_name)
            self.rs_images_dir = os.listdir(os.path.join(self.path_root, "image"))
            self.rs_images += [os.path.join(self.path_root, "image", img) for img in self.rs_images_dir]
            self.gt_images_dir = os.listdir(os.path.join(self.path_root, "label"))
            self.gt_images += [os.path.join(self.path_root, "label", img) for img in self.rs_images_dir]

    def __len__(self):
        return len(self.rs_images)

    def __getitem__(self, item):
        img = gdal.Open(self.rs_images[item])
        label = gdal.Open(self.gt_images[item])
        img = img.ReadAsArray().transpose(1, 2, 0)
        label = label.ReadAsArray()
        img = img / 255.0
        label = label / 255.0
        img = tfs.ToTensor()(img)
        label = tfs.ToTensor()(label)

        return img, label

        def __len__(self):
            return len(self.rs_images)
