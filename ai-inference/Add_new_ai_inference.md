# Add new AI Inference

We'll give a short tutorial how you can add your own AI model to the API.

> It should automatically update in the GeoNode UI after we push the new docker with your changes to the server.

## Status

* Docu: in progress

## Get started

First we make sure that your model file is a the right position (name and path)

1. Add your model file to the models folder for local testing
1. Add your model file to this folder [Download the models](https://cloud.dfki.de/owncloud/index.php/s/rQWsqfktx2drabb) to make it available for the server

Next we add your logic. Follow the steps or use the [setup helper](./create_new_inference_helper.py). The helper script only needs the model/folder name and creates the folder and inference file with all needed basics for you.

> Please use the underscore symbol to connect words in the folder name. -> use underscores '_' instead of hyphens '-' or spaces

1. Navigate to the folder [ai_inference/inference](./ai_inference/inference/)
1. Create a folder here with your model name and go into that folder (This name will be displayed in the GeoNode UI)
1. Create an *inference.py* file
1. take inspiration from the existing inference.py files

After you have the inference file you can fill it with your logic and add other modules to the folder as needed.

> There should be no need to change something in the other files outside your folder.

## Authors

Felix Becker <felix.becker@dfki.de>
