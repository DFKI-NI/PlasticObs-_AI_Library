from csv import writer
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from typing import List, Tuple, Any


def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate the model on a given dataset and return the average loss.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to run the evaluation on.

    Returns:
        float: Average loss over the dataset.
    """
    model.train()  # Set to train to still calculate loss
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images if image is not None]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets if t is not None]

            with torch.amp.autocast(device_type=device.type):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()
            count += 1

    avg_loss = total_loss / count if count > 0 else 0.0
    print(f"Averaged loss: {avg_loss:.4f}")
    return avg_loss


def validate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """
    Validate the model on a validation dataset.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the validation on.

    Returns:
        float: Validation loss.
    """
    print("Validating...")
    return evaluate(model, val_loader, device)


def test(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    Test the model on a test dataset.

    Args:
        model (torch.nn.Module): The model to test.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the test on.

    Returns:
        float: Test loss.
    """
    print("Testing...")
    return evaluate(model, test_loader, device)


def initialize_model(num_classes: int, weights: MaskRCNN_ResNet50_FPN_Weights, hidden_layer: int) -> torch.nn.Module:
    """
    Initialize a Mask R-CNN model with a custom head.

    Args:
        num_classes (int): Number of output classes.
        weights (MaskRCNN_ResNet50_FPN_Weights): Pretrained weights to use.
        hidden_layer (int): Number of hidden units in the mask predictor.

    Returns:
        torch.nn.Module: The initialized model.
    """
    model = maskrcnn_resnet50_fpn(weights=weights, progress=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels_mask, hidden_layer, num_classes)

    return model


def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) between two sets of boxes.

    Args:
        boxes1 (torch.Tensor): First set of bounding boxes.
        boxes2 (torch.Tensor): Second set of bounding boxes.

    Returns:
        torch.Tensor: IoU matrix.
    """
    return box_iou(boxes1, boxes2)


def calculate_metrics(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score based on predicted and ground truth boxes.

    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes.
        gt_boxes (torch.Tensor): Ground truth bounding boxes.
        iou_threshold (float, optional): IoU threshold to consider a match. Defaults to 0.5.

    Returns:
        Tuple[float, float, float]: Precision, recall, and F1 score.
    """
    ious = calculate_iou(pred_boxes, gt_boxes)
    matches = ious > iou_threshold
    true_positives = matches.sum().item()
    false_positives = len(pred_boxes) - true_positives
    false_negatives = max(0, len(gt_boxes) - true_positives)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def write_data(data: List[Any], path: str) -> None:
    """
    Write a list of data to a CSV file.

    Args:
        data (List[Any]): List of data to write.
        path (str): File path to the CSV file.
    """
    with open(path, 'a+', newline='') as new_row:
        csv_writer = writer(new_row)
        csv_writer.writerow(data)


def plot_confusion_matrix(cm: np.ndarray, row_names: List[str], col_names: List[str], title: str = 'Confusion Matrix', cmap: str = 'Blues') -> None:
    """
    Plot a confusion matrix using seaborn heatmap.

    Args:
        cm (np.ndarray): Confusion matrix.
        row_names (List[str]): Labels for the rows (true labels).
        col_names (List[str]): Labels for the columns (predicted labels).
        title (str, optional): Title of the plot. Defaults to 'Confusion Matrix'.
        cmap (str, optional): Color map to use. Defaults to 'Blues'.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=col_names, yticklabels=row_names)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def calculate_cm_metrics(confusion_matrix: List[List[int]]) -> Tuple[float, float, float, float]:
    """
    Calculate accuracy, precision, recall, and F1 score from a confusion matrix.

    Args:
        confusion_matrix (List[List[int]]): Confusion matrix.

    Returns:
        Tuple[float, float, float, float]: Accuracy, precision, recall, and F1 score.
    """
    cm = np.array(confusion_matrix)
    true_positives = np.diag(cm)
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.true_divide(true_positives, (true_positives + false_positives))
        recall = np.true_divide(true_positives, (true_positives + false_negatives))
        f1_score = 2 * np.true_divide(precision * recall, (precision + recall))

    accuracy = np.sum(true_positives) / np.sum(cm)

    precision = np.nan_to_num(precision).mean()
    recall = np.nan_to_num(recall).mean()
    f1_score = np.nan_to_num(f1_score).mean()

    return accuracy, precision, recall, f1_score


def load_labels(annFile: str) -> List[str]:
    """
    Load category labels from a COCO-style annotation JSON file.

    Args:
        annFile (str): Path to the annotation file.

    Returns:
        List[str]: List of category names.
    """
    with open(annFile) as f:
        data = json.load(f)
    categories = [d["name"] for d in data["categories"]]
    return categories
