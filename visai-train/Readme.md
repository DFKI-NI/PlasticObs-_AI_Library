# Training VIS AI Docker

## Get started

1. Create a folder with the .png files in the format (3200, 480)
1. Set the path (local path) to your images in the [docker compose](docker-compose.yml)
1. Override the [train.env](./train.env) file
    1. Set the parameters for training
1. Build the docker

This commnd will automatically run the container after building.

> You can check the [logs](#check-the-logs) to see the progress or error messages.

```bash
docker compose up --build
```

To run the docker you can use

```bash
docker compose up
```

## Check the logs

how do you check logs?

```bash
docker logs {CONTAINER ID}
```

## Common errors

It might be the case that your RAM isn't big enough. If so, you can reduce the batch size in the [train.env] (./train.env) file. We'd suggest a minimum of 4, though.

Another common mistake is that the path to your images isn't set up right. Just a heads-up: **Only** change the local path.

## Get model from docker

To get your model from the docker after training you get use the following command. This will copy your model to your curretn directory on your machine. To get the model name you can [check the logs](#check-the-logs).

```bash
docker cp {my_container}:/app/models/{your_model} .
```

## License

The example data and code in this repository and the subfolder is released under the BSD-3 license.

## Funding

Funded by the German Federal Ministry for the Environment, Nature Conservation, Nuclear Safety and Consumer Protection (BMUV) based on a resolution of the German Bundestag (Grant No. 67KI21014A).

## Authors/ maintainers

* Felix Becker, DFKI, <felix.becker@dfki.de>
