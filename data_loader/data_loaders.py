from torchvision import datasets, transforms
from base import BaseDataLoader
import json
import torch
import os
from torchvision.io import read_image
from PIL import Image
import numpy as np
import utils

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



class PN2PPDataSet(torch.utils.data.Dataset):
    def __init__(self, root, FOV, frequency):

        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "sequence"))))
        self.label = list(sorted(os.listdir(os.path.join(root, "label"))))
        self.FOV = FOV
        self.freqeuncy = frequency

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = read_image(imgs[idx])
        perspec_img = PN2PP(img, FOV, frequency)
        with open(self.label[idx], "r") as label_js:
            label = json.load(label_js)
        perspec_img = PN2PP_label(label, FOV, freqeuncy)
        return perspec_img, perspec_img
        

class PN2PPDataLoaer(BaseDataLoader):
    """
    Panorama data 2 perspective image DataLoader
    """
    def __init__(self, FOV, frequency, data_dir, batch_size =1 , shuffle=True, validation_split=0.0, num_workers=1):
        """
        frequency : sampling frequency in panorama images
                    ex) frequency : 36 -> sample perspective view with  10 degree interval
        """
        self.freqeuncy = frequency
        self.FOV = FOV
        self.dataset = PN2PPDataSet(data_dir, FOV, frequency)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers , collate_fn = utils.collate_fn)





def collate_fn_zp(batch): 
    return (zip(*batch))

class PseudoDatasetLoader(BaseDataLoader):
    """
    use data in dataset_pseudo directory
    there are only one pair in 인덕원/pair_data directory
    we will use Psuedo model which  have three stage .
    Modle's first stage  is pretrained model from torchvision 
    and It will use GT BB box data and pool from CNN feature map to ROI feature using pretrained torchvision CNN model
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = PseudoDataSet(self.data_dir, trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
class PseudoDataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.root = data_dir
        with open(os.path.join(data_dir,'pair_list.json'), 'r') as f:
            pair_list = json.load(f)
        self.pair_list = pair_list

    def __len__(self):
        return self.pair_list["Num_Pair"]
    def __getitem__(self, idx):
        t = self.pair_list

        pair_data_path = os.path.join(self.root ,self.pair_list["region"][0],'pair_data')
        
        pair_data_path2 = os.path.join(self.root, 'region2')
        i1_path = os.path.join(pair_data_path, 'I1.jpg')
        i1_label = os.path.join(pair_data_path, 'I1.json')
        i2_path = os.path.join(pair_data_path2, '1_37.6789_126.7546_10.0.jpg')
        i2_label = os.path.join(pair_data_path2, '1_37.6789_126.7546_10.0.json')
        with open(i1_label) as f:
            json_1 = json.load(f)
        with open(i2_label) as g:
            json_2 = json.load(g)

        obj_class = []
        for obj in json_1["shapes"]:
            if obj['label'] not in obj_class: obj_class.append(obj['label'])
        for obj in json_2["shapes"]:
            if obj['label'] not in obj_class: obj_class.append(obj['label'])
                


        I1_label ={}
        boxes1 = []
        labels1 = []
        for obj in json_1["shapes"]:
            labels1.append(obj_class.index(obj['label']))
            xyxy = obj['points']
            xyxy = np.array(xyxy)
            xmin = np.min(xyxy[:,0])
            xmax = np.max(xyxy[:,0])
            ymin = np.min(xyxy[:,1])
            ymax = np.max(xyxy[:,1])
            boxes1.append([xmin, ymin, xmax, ymax])
        I1_label["boxes"] = torch.as_tensor(boxes1)
        I1_label["labels"] = torch.as_tensor(labels1)

        I2_label = {}
        boxes2 = []
        labels2 = []
        for obj in json_2["shapes"]:
            labels2.append(obj_class.index(obj['label']))
            xyxy = obj['points']
            xyxy = np.array(xyxy)
            xmin = np.min(xyxy[:,0])
            xmax = np.max(xyxy[:,0])
            ymin = np.min(xyxy[:,1])
            ymax = np.max(xyxy[:,1])
            boxes2.append([xmin, ymin, xmax, ymax])
        I2_label["boxes"] = torch.as_tensor(boxes2)
        I2_label["labels"] = torch.as_tensor(labels2)
        i1 = self.transform(Image.open(i1_path).convert("RGB"))
        i2 = self.transform(Image.open(i2_path).convert("RGB"))

        breakpoint()

        # i1 ,i2= {"image": self.transform(Image.open(i1_path).convert("RGB")), "label": I1_label}, \
        # {"image": self.transform(Image.open(i2_path).convert("RGB")), "label": I2_label}

        return i1,I1_label, i2, I2_label

    