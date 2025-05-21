import os
import json
import random
import datetime
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from multiprocessing import freeze_support

from utils import test, validate
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights


# create data loader
class COCODataset(Dataset):
    """A custom torch dataset for COCO v1.0 annotations

    Args:
        Dataset (class): from torch.utils.data import Dataset
    """
    def __init__(self, root, annFile, transforms=None, resize=None):
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
            #print(f"Error loading image {path}: {e} ")
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
            img, target = transform(img, target)

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
    masks = masks.unsqueeze(1)  # Add channel dimension
    masks = torch.nn.functional.interpolate(masks.float(), size=size, mode="bilinear", align_corners=False).byte()
    masks = masks.squeeze(1)  # Remove channel dimension
    target["masks"] = masks

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


# Initialize model with pre-trained weights
def initialize_model(num_classes, weights, hidden_layer):
    """Initialize a model structure

    Args:
        num_classes (int): Number of classes for the model
        weights (_type_): Weights to initialize the model
        hidden_layer (int): Number of hidenn layers

    Returns:
        torch.nn.Module: The initialized model
    """
    model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get number of input features for the mask classifier
    in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels_mask, hidden_layer, num_classes)
    
    return model


def main():
    # select files
    root = 'D:/DATA/PlasticObsPlus/EOIR_AI/images/RGB_0.002'
    annFile = 'D:/DATA/PlasticObsPlus/EOIR_AI/annotations/annotations_4m/instances_default.json'
    
    # Training loop
    num_epochs = 10
    model_dir = 'models'
    
    # save model
    os.makedirs(model_dir, exist_ok=True)
    
    scaler = GradScaler()

    # load labels
    with open(annFile) as f:
        data = json.load(f)

    categories = []
    for d in data["categories"]:
        categories.append(d["name"])

    num_classes = len(categories)+1
    
    hidden_layer = 256
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    model = initialize_model(num_classes, weights, hidden_layer)

    # Assuming you have a list of all indices in your dataset
    dataset = COCODataset(root=root, annFile=annFile, transforms=get_transform(train=True), resize=resize)
    indices = list(range(len(dataset)))

    # Split indices for train, validation, and test sets
    train_indices, val_test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=42)

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # train
    # Move model to the appropriate device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model.to(device)

    # Define the optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0.0005)
    #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    model_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        i = 1
        for images, targets in train_loader:
            images = list(image.to(device) for image in images if image is not None)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets if t is not None]

            # Forward pass
            with torch.amp.autocast(device_type=device.type):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {losses.item():.4f}")
            i += 1

        # Update the learning rate
        lr_scheduler.step()
        print(f"Epoch {epoch+1} finished with loss: {losses.item():.4f}")

        # Validate the model
        val_loss = validate(model, val_loader, device)
        print(f"Validation Loss: {val_loss}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            model_name_short_run = f'{model_start_time}_Multiscale_Instance_Segmentation_best_loss.pt'.replace(':', '-').replace(' ', '_')
            model_name_short_run = os.path.join(model_dir, model_name_short_run)
            torch.save(model.state_dict(), model_name_short_run)

    print("Training completed!")
    # Test the model after training
    test_loss = test(model, test_loader, device)
    print(f"Test Loss: {test_loss}")
    
    model_name = f'{model_start_time}_Multiscale_Instance_Segmentation_epochs_{num_epochs}.pt'.replace(':', '-').replace(' ', '_')
    model_name = os.path.join(model_dir, model_name)
        
    torch.save(model.state_dict(), model_name)
    print("Model saved successfully!")
    print(f"Best Epoch was: {best_epoch} with validation loss: {best_loss}")


if __name__ == '__main__':
    freeze_support()
    main()
