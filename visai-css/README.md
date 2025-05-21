# VISAI Optimare

The repo for the deployment of the VIS AI on the aircraft from Optimare.

## Table of contents

1. [Getting started](#getting-started)
1. [Install dependencies for local use](#install-dependencies-for-local-usage)
1. [Docker](#docker)
1. [Documentation](#documentation)
    1. [env file](#env-file)
    1. [Dockerfile](#dockerfile)
    1. [IO descriptions](#io-descriptions)
    1. [Python scripts](#python-scripts)
1. [Authors and Acknowledgment](#authors-and-acknowledgment)
1. [License](#license)

## Project status

* create Docker/ Docker compose: done
* create running VIS AI: currently (3200, 480, 3) image size
* test UDP data transport: next steps/ hand over to Optimare
* work on documentation: done

ToDos:

* Train new model with new resolution

## Getting started

> See the [docker section](#docker) for info on the docker.

You can use the *main_thread.py* script for test runs.

```bash
python main_thread.py
```

## Install dependencies for local usage

As always, it is recommended to use a virtual environment.

> Please select a Pytorch version that is reasonable for your machine.

1. Install the other dependencies using the [requirements.txt](./requirements.txt) file.
1. Install [pytorch](https://pytorch.org/get-started/locally/) for the model inference.

```bash
pip install -r ./requirements.txt
```

```bash
python main_thread.py
```

## Docker

> Before running make sure that you have the correct port and hosts in the [host_ip.env](./host_ip.env) file.  
Additionally please check the [docker-compose.yml](./docker-compose.yml) file for the correct *network_mode* settings.

The **default** network_mode is:

```bash
network_mode: bridge
```

To build the docker, you can use this command

```bash
docker compose --env-file host_ip.env build
```

To run the docker image, you can use

```bash
docker compose up
```

> The docker currently runs the [main_thread.py](./main_thread.py) file.

## Documentation

### env file

For the env file we use the [host_ip.env](./host_ip.env) file.

In the env file you can configure:

* **HOSTNAME**: The IP address of the host machine.
  * Default: `192.168.144.44`

* **LOCAL_PORT**: The port number on the local machine where the application will run.
  * Default: `7799`

* **UNIQUE_VIEW_ID**: A unique identifier for the view.
  * Default: `2205`

* **VIEW_CHANNEL**: The channel used for viewing.
  * Default: `HRVIS`

* **VIS_IP**: The IP address for the visualization service.
  * Default: `192.168.0.105`

* **VIS_PORT**: The port number for the visualization service.
  * Default: `7777`

* **ANOMALIE_IP**: The IP address for the anomaly detection service.
  * Default: `192.168.0.105`

* **ANOMALIE_PORT**: The port number for the anomaly detection service.
  * Default: `7789`

* **MODEL_NAME**: The name of the VIS AI model file, which is in the models folder which is copied to the docker
  * Default: `torch_model_optimare_2024-03-21_15-58-01_epochs_10_batchsize_32_EmbeddingDim_35.pt`

### Dockerfile

We use a nvcr.io/nvidia/pytorch base image.

Currently we use PORT 7799 (*host_ip.env*) for incoming traffic from the VIS Line scanner. For outgoing we didn't bind a specific port.

### IO descriptions

The script expects a running *VIS line scanner* to send data. Additionally we need a running instance to receive anomalies.

#### Anomalies

The anomalies are constructed as seen in the example:

> Anomaly(id, line_id, pixel_id, priority, probability)

* ID: The ID is a running number in the current mission.
* Line ID: The Line ID is the number of the line send by the VIS line scanner.
* Pixel ID: The Pixel ID is the number of the pixel in a line.
* Priority: The priority is based on the position and probability of an anomaly. *Defined: [0, n]*, where 0 is the most important.
* Probability: The probability describes how certain the model found a anomaly in the images. *Defined: [0, 1]*, where 1 is the most certain.

### Python scripts

Here we have a short description of the modules, classes and functions.

#### inference

* `VISAIModel`: Wrapper class for VIS AI Model, Variational Autoencoder
  * `__init__`: Initialize the class
  * `_load_model`: Load a variational autoencoder (vae) from a pytorch state model (.pt/.pth)
  * `inference`: Runs the VAE model and filters result with alpha channel

#### main

System settings

* define logging
* `load_env_file(file_path)`: Loads and reads env file

UDPCollector

* `UDPCollector`: Class for handeling the UDP data
* `__init__(self, model_name, fullRes, num_of_lines, Anomalie_IpPort)`: initializes UDP collector class
* `decode_data(self, data)`: Decoding of the byte string from the VIS Line Scanner
* `collect_udp_data(self)`: Collect udp data from socket
* `evaluate_data(self, imgBuffer, alpha, line_ids)`: Starts the model inference and calls the anomalie sender

#### VAE model

Here we set up the structure for the VAE model.

Custom Loss

* `LogCoshLoss(nn.Module)`: Custom Loss class
* `__init__(self)`: Initialize Custom Loss class
* `forward(self, y_t, y_prime_t)`: Calculate forward pass of the loss function

Loss calculations

* `vae_gaussian_kl_loss(mu, logvar)`: Calculate the kl loss of the autoencoder
* `reconstruction_loss(x_reconstructed, x)`: Calculate the reconstruction loss
* `vae_loss(y_pred, y_true)`: Calculate the losses for the VAE

Sampling

* `Sampling(nn.Module)`: This class will be used in the encoder for sampling in the latent space
* `forward(self, z_mean, z_log_var)`: Forward pass for the Sampling class

Encoder

* `Encoder(nn.Module)`: The encoder for the VAE
* `__init__(self, image_size, embedding_dim)`: Initializes the encoder class
* `forward(self, x)`: Forward pass for the VAE

Decoder

* `Decoder(nn.Module)`: The decoder class for an AutoEncoder
* `__init__(self, embedding_dim, shape_before_flattening)`: Initializes the decoder class
* `forward(self, x):`: Forward pass for the VAE

VAE Model

* `VAE(nn.Module)`: VAE Model
* `__init__(self, encoder, decoder)`: Initialize the VAE Model class
* `forward(self, x)`: Forward pass for the VAE

Prediction functions

* `predict_single(model, test_image, DEVICE)`: Predict one image w/ a VAE Model
* `model_prediction(vae_model, image_name, threshold, img_shape, DEVICE)`: Make model prediction and anomaly filtering

#### VIS Anomaly Sender

Anomaly class

* `__init__(self, id, line_id, pixel_id, priority, probability)`: Build an anomaly object

AnomalySender class

* `__init__`: Initalize anomaly sender
* `_send_anomalies`: Sends the anomalies to the anomaly receiver
* `get_anomalies_from_medoids`: handles the the conversion from medoids list to Anomalies
* `get_prio`: Calculate priority (0 or 1 is highest) based on the position and probapility

## License

The example data and code in this repository is released under the BSD-3 license.

## Funding

Funded by the German Federal Ministry for the Environment, Nature Conservation, Nuclear Safety and Consumer Protection (BMUV) based on a resolution of the German Bundestag (Grant No. 67KI21014A).

## Authors and acknowledgment

**Felix Becker / DFKI** - Development and Documentation

Special thanks to:

* **Sören Schweigert / Optimare** - for providing test programs in java
