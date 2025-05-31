# PlasticObs+ AI Library

The AI libary (german: KI-Bibliothek) is a collection of train, inference and use cases for the AI models that have been developed in the PlasticObs_plus (PO+) project.

The BMUV-funded PlasticObs_plus project (2022-2025) developed an AI-based remote sensing system for detecting plastic waste in waters. It combines RGB line scanning with real-time AI (VIS AI) and a multispectral high-resolution sensor (EOIR AI) to identify plastic from the air. VIS AI analyzes data in real-time, prioritizing suspicious areas, while EOIR AI performs detailed segmentation. Tests in Brazil, coastal regions, and festivals confirmed its adaptability. The open-source GeoNode platform integrates these models, enabling worldwide use and regional customization. The consortium (DFKI, everwave, Jade University, Optimare) provided interdisciplinary expertise for practical implementation. All models and datasets are publicly available to authorities, NGOs, and researchers. The system bridges satellite overview imaging and drone-based detailed analysis, creating efficient monitoring strategies for maritime regions.
Funding Reference 67KI21014A

## Table of contents

1. [Technical terms](#technical-terms)
1. [Subfolder](#subfolder)
1. [Related Repository](#related-repository)
1. [License](#license)
1. [Funding](#funding)
1. [Authors and maintainers](#authors-maintainers)

## Technical terms

| Term       | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| CSS        | Candidate Selection System                                                  |
| EOIR AI    | An AI developed for an electronic optic infrared camera and drone images    |
| VIS        | Visual spectrum                                                             |
| VIS AI     | An AI developed for a VIS line scanner                                      |

## Subfolder

In each subfolder we present a aprt of the results and use cases of the PO+ project.

### [ai-inference](./ai-inference/)

A docker based FastAPI with an AI inference for a [custom GeoNode implementation](https://github.com/52North/plasticobs-geonode). Some of the models used in this AI API are presented in the other subfolder as well.

### [eoir-ai](./eoir-ai/)

Scripts and other content for the EOIR AI which is used to detect waste in images with an instance segmentation and bounding boxes.

### [feedback-loop](./feedback-loop/)

At the moment it is only a [Readme](./feedback-loop/Readme.md) with the theoretic background on the feedback loop.

### [visai-css](./visai-css/)

VIS AI in combination with a candidate selection system (CSS)

docker based

### [visai-train](./visai-train/)

A docker to train a model for the VIS AI

## Related Repository

If you're working with annotation datasets and looking for tools to adapt them across different formats or tasks, make sure to check out our companion repository: [DFKI-NI/Adapting_Annotation_Datasets](https://github.com/DFKI-NI/Adapting_Annotation_Datasets)
This repository provides tools and scripts that adapt various annotation datasets into formats directly usable by our model. It streamlines preprocessing and maintains consistency across different data sources, making it an integral component of the EOIR AI data pipeline.

The dataset used for training and evaluating our AI model is publicly available on Zenodo: [Multiscale_Waste_PlasticObs_plus](https://zenodo.org/records/15126023). This dataset is designed to work seamlessly with both our model and the tools provided in the Adapting_Annotation_Datasets repository. It includes preprocessed and raw annotation data, ensuring reproducibility and ease of experimentation.

## License

The example data and code in this repository, including all contents in the subfolders, are released under the **BSD 3-Clause License**.

Please note that the [ai-inference](./ai-inference/) component is licensed separately under the **MIT License**. Be sure to review the respective license files for details.

## Funding

Funded by the German Federal Ministry for the Environment, Nature Conservation, Nuclear Safety and Consumer Protection (BMUV) based on a resolution of the German Bundestag (Grant No. 67KI21014A).

## Authors maintainers

* Felix Becker, DFKI, <felix.becker@dfki.de>
* Robert Rettig, DFKI, <robert.rettig@dfki.de>
