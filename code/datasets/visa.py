import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv

from .self_sup_tasks import patch_ex



CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2','pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']


class VisaDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        image_size = 224
        print(f'[VisaDataset] image_size={image_size}')
        self.transform = transforms.Resize(
                                (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
                            )
        
        self.norm_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        )
        
        datas_csv_path = os.path.join(root_dir, 'split_csv', '1cls.csv')

        self.paths = []
        self.x = []

        with open(datas_csv_path, 'r') as file:
            reader = csv.reader(file)

            for row in reader:
                if row[1] == 'train' and row[0] in CLASS_NAMES:
                    file_path = os.path.join(root_dir, row[3])
                    self.paths.append(file_path)
                    self.x.append(self.transform(Image.open(file_path).convert('RGB')))

        
        self.prev_idx = np.random.randint(len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path, x = self.paths[index], self.x[index]
        class_name = img_path.split('/')[-5]

        self_sup_args={'width_bounds_pct': ((0.03, 0.4), (0.03, 0.4)),
                    'intensity_logistic_params': (1/12, 24),
                    'num_patches': 2,
                    'min_object_pct': 0,
                    'min_overlap_pct': 0.25,
                    'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                    'shift':True, 
                    'same':False, 
                    'mode':cv2.NORMAL_CLONE, 
                    'label_mode':'logistic-intensity',
                    'skip_background': None,
                    'resize_bounds': (.5, 2)
                    }

        x = np.asarray(x)
        origin = x

        p = self.x[self.prev_idx]
        if self.transform is not None:
            p = self.transform(p)
        p = np.asarray(p)    
        x, mask, centers = patch_ex(x, p, **self_sup_args)
        mask = torch.tensor(mask[None, ..., 0]).float()
        self.prev_idx = index
        

        origin = self.norm_transform(origin)
        x = self.norm_transform(x)


        return origin, x, class_name, mask, img_path



    def collate(self, instances):

        images = []
        class_names = []
        masks = []
        img_paths = []
        for instance in instances:
            images.append(instance[0])
            class_names.append(instance[2])
            masks.append(torch.zeros_like(instance[3]))
            img_paths.append(instance[4])

            images.append(instance[1])
            class_names.append(instance[2])
            masks.append(instance[3])
            img_paths.append(instance[4])


        return dict(
            images=images,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths
        )
