import os
import numpy as np
import tifffile as tiff
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as F

# create data loader
class COCODatasetMultispectral(Dataset):
    def __init__(self, root, annFile, transforms=None, resize=None):
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
            img = tiff.imread(os.path.join(self.root, path))
            if img.shape[-1] != 5:
                raise ValueError(f"Expected a 5-channel image, but got {img.shape[-1]} channels.")
            img = handle_nans(img)
            img = torch.from_numpy(img).permute(2, 0, 1).float()  # Convert to tensor and rearrange dimensions
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return self.__getitem__((index + 1) % len(self.ids))

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
    
def handle_nans(img):
    """handles tiff images with NaNs and replaces them with 0.0

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """
    if np.isnan(img).any():
        #print("Image contains NaNs. Handling NaNs...")
        img = np.nan_to_num(img, nan=0.0)  # Replace NaNs with 0.0 or another value
    return img
