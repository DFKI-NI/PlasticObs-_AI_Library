# EOIR AI

The EOIR AI is designed to detect and classify plastic and other types of waste in images.

## Table of Contents

1. [Getting started](#getting-started)
1. [Starting Training](#starting-training)
1. [Python scripts](#python-scripts)
1. [Folders](#folders)
1. [Authors](#authors-maintainers)

## Getting started

To get started to create your own models or checkout our inference scripts you first need to create an virtual enivorment and install the requirements.

```sh
pip install -r requirements.txt
```

After that you need to install your preferrred pytorch installation.

### Harmonizing and Combining

For harmonizing and combining your annotation files you can checkout our [Github repo](https://github.com/DFKI-NI/Adapting_Annotation_Datasets) on that topic.

## Starting Training

We focus on instance segmentation with our models.

We have two different train scripts.

1. For standard RGB images (jpg, png)
2. For multispectral images (5 channels/ tiff)

The first step is always to prepare your data in the COCO format. And adjust the path names in the corresponding train script.

### Traioning with RGB images

1. Navigate to the main function
1. Replace the *root* entry with your images path
1. Replace the *annFile* with your annotation file
1. Adjust the number of epochs *num_epochs*
1. Set a model directory to save your models *model_dir*

```sh
python train_instance_segmentation.py
```

### Training with tiff images

1. Navigate to the main function
1. Replace the *root* entry with your images path
1. Replace the *annFile* with your annotation file
1. In this case you need to search a bit to find the other parameters
1. Adjust the number of epochs *num_epochs*
1. Set a model directory to save your models *model_dir*

```sh
python train_multispectral_instance_segmentation.py
```

## Python scripts

### dataloader

A module where all stuff related to data loading for the RGB model is present.

* `class COCODatset(Dataset)`: A custom torch dataset for COCO v1.0 annotations
  * Standard PyTorch dataset methods are used.
* `resize(img, target, size=(1000, 1000))`: Resize images and targets to match input sizes for the model
* `get_transform(train)`: Define a transformation for image data
* `transform(img, target)`: Transform the data by Flipping it randomly
* `collate_fn(batch)`:  Custom collate function for data loading.

### inference

A module to quickly load models for inference. used for both the RGB and the multispectral models. All classes have an init method.

* `class MaskRcnnInference()`: Prepare a RGB model for inference
* `class CustomGeneralizedRCNNTransform(GeneralizedRCNNTransform)`: Custom transformation class for Generalized RCNN models to handle images with 5 channels.
* `class MaskRcnnInference5Channels()`: Prepare a multispectral model for inference

### multispectral_dataloader

A module where all stuff related to data loading for the multispectral model is present.

* `COCODatasetMultispectral(Dataset)`: A custom torch dataset for COCO v1.0 annotations
  * Standard PyTorch dataset methods are used.
* `handle_nans(img)`: handles tiff images with NaNs and replaces them with 0.0

### preprocessing

Preprocessing for the inference module

* `resize(img, size=(1000, 1000))`: Resize an image
* `get_transform(train)`: Function to create transform object

### test_inst_seg_models

A script to validate and test your models.

* `calculate_iou(boxes1, boxes2)`: Calculate the Intersection over Union (IoU) between two sets of bounding boxes.
* `calculate_metrics(pred_boxes, gt_boxes, iou_threshold=0.5)`: Calculate precision, recall, and F1 score based on predicted and ground truth bounding boxes.
* `main(model_path: str, annFile: str, root: str, num_classes: int, device: torch.device)`: main loop for model and data setup

### train instance segmentation

Training script for standard RGB images (JPG, PNG). This script was developed first, and we later moved the [dataloader](#dataloader) into its own module. It will be removed from this file in the future.

* `class COCODatset(Dataset)`: A custom torch dataset for COCO v1.0 annotations
  * Standard PyTorch dataset methods are used.
* `resize(img, target, size=(1000, 1000))`: Resize images and targets to match input sizes for the model
* `get_transform(train)`: Define a transformation for image data
* `transform(img, target)`: Transform the data by Flipping it randomly
* `collate_fn(batch)`:  Custom collate function for data loading.
* `initialize_model(num_classes, weights, hidden_layer)`: Initialize a model structure
* `main()`: main loop

### train multispectral instance segmentation

Training script for a model that can work with multispectral images (5 channels/ tiff).

* `class CustomGeneralizedRCNNTransform(GeneralizedRCNNTransform)`: Custom transformation class for Generalized RCNN models to handle images with 5 channels.
* `class CustomTransform`: Transfrom data
* `collate_fn(batch)`:  Custom collate function for data loading.
* `initialize_model(num_classes, weights, hidden_layer, num_channels)`: Initilaize a model structure
* `main()`: main loop

### utils

* `evaluate(model, data_loader, device)`: Evaluate the model on a dataset and return the average loss.
* `validate(model, val_loader, device)`: Run evaluation on the validation dataset.
* `test(model, test_loader, device)`: Run evaluation on the test dataset.
* `initialize_model(num_classes, weights, hidden_layer)`: Initialize a Mask R-CNN model with a custom head for classification and segmentation.
* `calculate_iou(boxes1, boxes2)`: Calculate the Intersection over Union (IoU) between two sets of bounding boxes.
* `calculate_metrics(pred_boxes, gt_boxes, iou_threshold=0.5)`: Calculate precision, recall, and F1 score based on predicted and ground truth bounding boxes.
* `write_data(data, path)`: Write a list of data entries to a CSV file.
* `plot_confusion_matrix(cm, row_names, col_names, title='Confusion Matrix', cmap='Blues')`: Plot a confusion matrix using a seaborn heatmap.
* `calculate_cm_metrics(confusion_matrix)`: Calculate accuracy, precision, recall, and F1 score from a confusion matrix.
* `load_labels(annFile)`: Load category labels from a COCO-style annotation JSON file.

## Folders

### models

Here you can save your models.

## License

The example data and code in this repository is released under the BSD-3 license.

## Funding

Funded by the German Federal Ministry for the Environment, Nature Conservation, Nuclear Safety and Consumer Protection (BMUV) based on a resolution of the German Bundestag (Grant No. 67KI21014A).

## Authors/ maintainers

Felix Becker, DFKI
