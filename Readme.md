# PlasticObs+ AI Library

The AI libary (german: KI-Bibliothek) is a collection of train, inference and use cases for the AI models that have been developed in the PlasticObsplus (PO+) project.

*General info PO+*

## Technical terms

| Term       | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| CSS        | Candidate Selection System                                                  |
| EOIR AI    | An AI developed for an electronic optic infrared camera and drone images    |
| VIS        | Visual spectrum                                                             |
| VIS AI     | An AI developed for a VIS line scanner                                      |

## [ai-inference](./ai-inference/)

A docker based FastAPI with an AI inference for a [custom GeoNode implementation](https://github.com/52North/plasticobs-geonode). Some of the models used in this AI API are presented in the other subfolder as well.

## [eoir-ai](./eoir-ai/)

Scripts and other content for the EOIR AI which is used to detect waste in images with an instance segmentation and bounding boxes.

## [feedback-loop](./feedback-loop/)

At the moment it is only a [Readme](./feedback-loop/Readme.md) with the theoretic background on the feedback loop.

## [visai-css](./visai-css/)

VIS AI in combination with a candidate selection system (CSS)

docker based

## [visai-train](./visai-train/)

A docker to train a model for the VIS AI

## License

The example data and code in this repository and the subfolder is released under the BSD-3 license.

## Funding

Funded by the German Federal Ministry for the Environment, Nature Conservation, Nuclear Safety and Consumer Protection (BMUV) based on a resolution of the German Bundestag (Grant No. 67KI21014A).

## Authors/ maintainers

* Felix Becker, DFKI, <felix.becker@dfki.de>
* Robert Rettig, DFKI, <robert.rettig@dfki.de>
