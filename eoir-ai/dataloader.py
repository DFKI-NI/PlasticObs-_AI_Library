import os
import random
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as F


# create data loader
class COCODataset(Dataset):
    """A custom torch dataset for COCO v1.0 annotations

    Args:
        Dataset (class): from torch.utils.data import Dataset
    """
    def __init__(self, root: str, annFile: str, transforms=None, resize=None):
        """Initilaize custom COCO datset

        Args:
            root (str): folder path to ypur images
            annFile (str): path and name to your annotation file
            transforms (_type_, optional): A transforms function for the data. Defaults to None.
            resize (_type_, optional): A resize function for the data. Defaults to None.
        """
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.resize = resize

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        try:
            img = Image.open(os.path.join(self.root, path)).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e} ")
            return self.__getitem__((index + 1) % len(self.ids))

        num_objs = len(anns)
        boxes = []
        labels = []
        masks = []

        for ann in anns:
            xmin, ymin, width, height = ann['bbox']
            if width <= 0 or height <= 0:
                continue  # Skip invalid annotations
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])
            masks.append(coco.annToMask(ann))

        if len(boxes) == 0:
            # Skip images without valid annotations
            return self.__getitem__((index + 1) % len(self.ids))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)  # Convert list of numpy arrays to a single numpy array

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks

        if self.resize is not None:
            img, target = self.resize(img, target)

        if self.transforms is not None:
            img = self.transforms(img)
            #img, target = transform(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

def resize(img, target, size=(1000, 1000)):
    """Resize images and targets to match input sizes for the model

    Args:
        img (_type_): img to be resized
        target (_type_): target information to be resized
        size (tuple, optional): Resize size of the data for the model. Defaults to (1000, 1000).

    Returns:
        _type_: resized img and target
    """
    # Resize image
    w, h = img.size
    img = img.resize(size, Image.Resampling.BILINEAR)
    
    # Resize bounding boxes
    scale_x = size[0] / w
    scale_y = size[1] / h

    boxes = target["boxes"]
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    target["boxes"] = boxes

    # Resize masks
    masks = target["masks"]
    resized_masks = []
    for mask in masks:
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        mask = F.resize(mask, size, interpolation=F.InterpolationMode.NEAREST)
        mask = mask.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        resized_masks.append(mask)
    target["masks"] = torch.stack(resized_masks)  # Stack masks into a single tensor

    return img, target

def get_transform(train):
    """Define a transformation for image data

    Args:
        train (boolean): pass the info for train or not train

    Returns:
        torch.tensor: returns an array torch tensor
    """
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def transform(img, target):
    """Transform the data by Flipping it randomly

    Args:
        img (_type_): img to transform
        target (_type_): target information to transform

    Returns:
        _type_: _description_
    """
    # Apply random horizontal flip
    if random.random() > 0.5:
        img = F.hflip(img)
        img_width = img.size(2)
        target["boxes"][:, [0, 2]] = img_width - target["boxes"][:, [2, 0]]
        target["masks"] = target["masks"].flip(-1)
    return img, target

def collate_fn(batch):
    """Custom collate function for data loading.

    Args:
        batch (List[Tuple[Any, ...]]): A list of data samples, where each sample is a tuple.

    Returns:
        Tuple[List[Any], ...]: A tuple where each element is a list containing
            the respective elements from each sample in the batch
    """
    return tuple(zip(*batch))
