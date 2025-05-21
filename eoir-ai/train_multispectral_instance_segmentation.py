# libraries
import os
import json
import datetime
from multiprocessing import freeze_support
from sklearn.model_selection import train_test_split

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

from utils import test, validate
from multispectral_dataloader import COCODatasetMultispectral

# TODO: Add logger for the training

# Define a custom transform to include ToTensor and Normalize
class CustomGeneralizedRCNNTransform(GeneralizedRCNNTransform):
    """Custom transformation class for Generalized RCNN models to handle images with 5 channels.

    Args:
        min_size (int): Minimum size of the image after resizing.
        max_size (int): Maximum size of the image after resizing.
        image_mean (list): List of mean values for each channel used for normalization.
        image_std (list): List of standard deviation values for each channel used for normalization.

    Attributes:
        image_mean (torch.Tensor): Tensor of mean values for each channel.
        image_std (torch.Tensor): Tensor of standard deviation values for each channel
    """
    def __init__(self, min_size, max_size, image_mean, image_std):
        # Convert image_mean and image_std to tensors
        image_mean = torch.tensor(image_mean)
        image_std = torch.tensor(image_std)
        super().__init__(min_size, max_size, image_mean, image_std)

    def normalize(self, image):
        """
        Normalizes the input image using the mean and standard deviation values.
        Ensures that the mean and std tensors are on the same device as the image tensor.

        Args:
            image (torch.Tensor): Input image tensor to be normalized.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        device = image.device
        image_mean = self.image_mean.to(device)
        image_std = self.image_std.to(device)
        # Normalize the image with 5 channels
        return (image - image_mean[:, None, None]) / image_std[:, None, None]


# Define a custom transform to include ToTensor and Normalize
class CustomTransform:
    """Custom transform class
    """
    def __init__(self, resize_size):
        self.resize = transforms.Resize(resize_size)
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456])
        self.std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224])

    def __call__(self, image):
        # Print the original shape of the image
        #print("Original image shape:", image.size())
        
        # Resize the image
        image = self.resize(image)
        
        # Print the shape after resizing
        #print("Resized image shape:", image.size())
        
        # Normalize the image
        image = (image - self.mean.view(5, 1, 1)) / self.std.view(5, 1, 1)
        
        # Print the shape after normalization
       #print("Normalized image shape:", image.size())
        
        return image

   
def collate_fn(batch):
    return tuple(zip(*batch))

def initialize_model(num_classes, weights, hidden_layer, num_channels):
    """Initialize a model structure

    Args:
        num_classes (int): Number of classes for the model
        weights (_type_): Weights to initialize the model
        hidden_layer (int): Number of hidenn layers
        num_channels (int): Number of input channels

    Returns:
        torch.nn.Module: The initialized model
    """
    # Load a pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    
    # Modify the input layer to accept more channels
    # Get the original conv1 layer
    original_conv1 = model.backbone.body.conv1
    # Create a new conv1 layer with the desired number of input channels
    model.backbone.body.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Update the weights of the new input layer (optional)
    torch.nn.init.kaiming_normal_(model.backbone.body.conv1.weight, mode='fan_out', nonlinearity='relu')
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get number of input features for the mask classifier
    in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels_mask, hidden_layer, num_classes)
    
    # Replace the default transform with the custom transform
    model.transform = CustomGeneralizedRCNNTransform(
        min_size=800,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406, 0.485, 0.456],  # Update with appropriate mean values for 5 channels
        image_std=[0.229, 0.224, 0.225, 0.229, 0.224]    # Update with appropriate std values for 5 channels
    )
    
    return model

def main():
    # select files
    root = 'D:/DATA/PlasticObsPlus/EOIR_AI/Spot_Multispectral/Spot_0.00898'
    annFile = 'D:/DATA/PlasticObsPlus/EOIR_AI/Spot_Multispectral/annotations/annotations_20m/instances_default.json'
    
    scaler = GradScaler()
    
    # load labels
    with open(annFile) as f:
        data = json.load(f)
    
    categories = []
    for d in data["categories"]:
        categories.append(d["name"])

    num_channels = 5 # number of input channels
    num_classes = len(categories)+1
    hidden_layer = 256
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    model = initialize_model(num_classes, weights, hidden_layer, num_channels)
    
    # Define transform
    resize_size = (1000, 1000)
    transform = CustomTransform(resize_size)

    # Assuming you have a list of all indices in your dataset
    dataset = COCODatasetMultispectral(root=root, annFile=annFile, transforms=transform, resize=None)
    indices = list(range(len(dataset)))

    # Split indices for train, validation, and test sets
    train_indices, val_test_indices = train_test_split(indices, test_size=0.3, random_state=42)
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
    
    # Training loop
    num_epochs = 20

    model_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # save model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
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
            model_name_short_run = f'{model_start_time}_Multispectral_{num_channels}_Instance_Segmentation_best_loss.pt'.replace(':', '-').replace(' ', '_')
            model_name_short_run = os.path.join(model_dir, model_name_short_run)
            torch.save(model.state_dict(), model_name_short_run)

    print("Training completed!")
    # Test the model after training
    test_loss = test(model, test_loader, device)
    print(f"Test Loss: {test_loss}")
    
    model_name = f'{model_start_time}_Multispectral_{num_channels}_Instance_Segmentation_epochs_{num_epochs}.pt'.replace(':', '-').replace(' ', '_')
    model_name = os.path.join(model_dir, model_name)
        
    torch.save(model.state_dict(), model_name)
    print("Model saved successfully!")
    print(f"Best Epoch was: {best_epoch} with validation loss: {best_loss}")

if __name__ == '__main__':
    freeze_support()
    main()
