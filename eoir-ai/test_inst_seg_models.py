import os
import sys
import json
from typing import Tuple
from multiprocessing import freeze_support

# libraries
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from inference import MaskRcnnInference
from dataloader import COCODataset, get_transform, resize, collate_fn


# Function to calculate IoU
def calculate_iou(boxes1, boxes2):
    """
    Calculate the Intersection over Union (IoU) between two sets of boxes.

    Args:
        boxes1 (torch.Tensor): First set of bounding boxes.
        boxes2 (torch.Tensor): Second set of bounding boxes.

    Returns:
        torch.Tensor: IoU matrix.
    """
    return box_iou(boxes1, boxes2)


# Function to calculate precision, recall, and F1 score
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
    false_negatives = len(gt_boxes) - true_positives
    
    # Ensure false negatives are not negative
    false_negatives = max(0, false_negatives)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


def main(model_path: str, annFile: str, root: str, num_classes: int, device: torch.device):
    """Main loop for running the test on the model

    Args:
        model_path (str): Path/name of the model -> .pt file
        annFile (str): Path/name of the annotation  file
        root (str): Path to the images
        num_classes (int): Number of classes in the model
        device (torch.device): torch device where the model and data should be evaluated
    """
    inference = MaskRcnnInference(model_path=model_path, num_classes=num_classes)
    model = inference.model
    model.train()

    dataset = COCODataset(root=root, annFile=annFile, transforms=get_transform(train=False), resize=resize)

    test_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Validation loop
    model.eval()
    all_precisions = []
    all_recalls = []
    all_f1_scores = []

    for images, targets in test_loader:
        images = list(image.to(device) for image in images if image is not None)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets if t is not None]

        with torch.no_grad():
            predictions = model(images)

        for i, prediction in enumerate(predictions):
            pred_boxes = prediction['boxes']
            gt_boxes = targets[i]['boxes']

            precision, recall, f1_score = calculate_metrics(pred_boxes, gt_boxes)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1_score)
            
    # Calculate average precision, recall, and F1 score
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1_score = sum(all_f1_scores) / len(all_f1_scores)

    print(f"Average Precision: {avg_precision:.4f},\
        Average Recall: {avg_recall:.4f},\
        Average F1 Score: {avg_f1_score:.4f}")


if __name__ == '__main__':
    freeze_support()
    
    # define parameters
    model = './models/2024-12-10_16-28-33_Multiscale_Instance_Segmentation_best_loss.pt'
    annotations = 'D:/DATA/PlasticObsPlus/EOIR_AI/Materialtypes/instances_default_corr_materialtypes.json'
    images = 'D:/DATA/PlasticObsPlus/EOIR_AI/images'
    
    # load labels
    with open(annotations) as f:
        data = json.load(f)

    categories = []
    for d in data["categories"]:
        categories.append(d["name"])

    num_classes = len(categories)+1

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    main(model, annotations, images, num_classes, device)

