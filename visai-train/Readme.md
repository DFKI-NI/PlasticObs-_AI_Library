# Training VIS AI Docker

## Get started

1. Create a folder with the .png files in the format (3200, 480)
1. Override the [train.env](./train.env) file
    1. Set the path to your images
    1. Set the number of epochs
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

## Authors

Felix Becker <felix.becker@dfki.de>
