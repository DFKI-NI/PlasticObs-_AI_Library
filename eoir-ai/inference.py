import numpy as np
from io import BytesIO
from PIL import Image

import torch
import torchvision.transforms.functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

import preprocessing


class MaskRcnnInference():
    def __init__(self, proba_threshold=0.5, score_threshold=0.75,
                weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, num_classes=50,
                model_path = "./models/2024-09-24_13-30-51_Instance_Segmentation_best_loss.pt"):
        """Prepare a RGB model for inference

        Args:
            proba_threshold (float, optional): Probapility threshold for the model results. Defaults to 0.5.
            score_threshold (float, optional): Score threshold for the model results. Defaults to 0.75.
            weights (_type_, optional): Weights for initializing the model. Defaults to MaskRCNN_ResNet50_FPN_Weights.DEFAULT.
            num_classes (int, optional): Number of classes for the model. Defaults to 50.
            model_path (str, optional): Path to the statedict of a trained model. Defaults to "./models/2024-09-24_13-30-51_Instance_Segmentation_best_loss.pt".
        """
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.proba_threshold = proba_threshold
        self.score_threshold = score_threshold
        model = maskrcnn_resnet50_fpn(weights=weights)

        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        # Replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.DEVICE, weights_only=True))
        self.model = model.eval()
        self.model.to(self.DEVICE)

    def do_predict(self, input_file):
        """DEPRECATED: A method to do predictions, 

        Args:
            input_file (str): name of an image file

        Returns:
            np.array: image array with bounding boxes drawn on the detected objects
        """
        with Image.open(input_file).convert("RGB") as image:
            #trans_image = self.transforms(image).unsqueeze(0)
            inference_transform = preprocessing.get_transform(train=False)

            # Resize images and targets
            resized_image = preprocessing.resize(image)
            
            original_width, original_height = image.size
            resized_width, resized_height = resized_image.size

            # Calculate scaling factors
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height

            # Apply additional transformations
            trans_image = inference_transform(resized_image).unsqueeze(0).to(self.DEVICE)
            
            output = self.model(trans_image)  # type: ignore

            out = output[0]
            bool_masks = out['masks'][out['scores'] > self.score_threshold] > self.proba_threshold
            # print(f"shape = {bool_masks.shape}, dtype = {bool_masks.dtype}")

            # There's an extra dimension (1) to the masks. We need to remove it
            bool_masks = bool_masks.squeeze(1)

            image_tensor = F.pil_to_tensor(image)
            resized_bool_masks = torch.nn.functional.interpolate(bool_masks.unsqueeze(1).float(), size=image_tensor.shape[1:], mode='nearest').squeeze(1).bool()
            drawing = draw_segmentation_masks(image_tensor, resized_bool_masks, alpha=0.5)
            
            boxes = out["boxes"][out["scores"] > self.score_threshold]
            resized_boxes = boxes.clone()
            resized_boxes[:, [0, 2]] /= scale_x  # Scale x coordinates
            resized_boxes[:, [1, 3]] /= scale_y  # Scale y coordinates
            labels = out["labels"][out["scores"] > self.score_threshold]
            labels = [str(label) for label in labels]
            
            drawing_with_boxes = draw_bounding_boxes(drawing, resized_boxes, labels=labels, colors="red", width=8)

            img = drawing_with_boxes.detach().cpu()
            img = F.to_pil_image(img)

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

        with Image.open(buffer) as image_pil:
            return np.array(image_pil)


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


class MaskRcnnInference5Channels():
    def __init__(self, proba_threshold=0.5, score_threshold=0.75,
                weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, num_classes=50,
                model_path = "./models/2024-09-24_13-30-51_Instance_Segmentation_best_loss.pt"):
        # comment
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.proba_threshold = proba_threshold
        self.score_threshold = score_threshold
        hidden_layer = 256
        num_channels = 5

        model = maskrcnn_resnet50_fpn(weights=weights)

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

        model.load_state_dict(torch.load(model_path, map_location=self.DEVICE, weights_only=True))
        self.model = model.eval()
        self.model.to(self.DEVICE)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    input_file = "test_cases/Plot_4_4m_RGB.png"
    parameters = ""
    inference = MaskRcnnInference()
    result = inference.do_predict(input_file)
    plt.imshow(np.asarray(result))
    plt.show()
