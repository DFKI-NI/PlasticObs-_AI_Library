# libraries
import os
import ast
import sys
import logging
import datetime
import numpy as np
from PIL import Image

# libraries
import torch
import torch.optim as optim
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
from torchvision import transforms

from vae_model import Decoder, Encoder, VAE, vae_loss


# set system settings
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to read .env file and set environment variables
def load_env_file(file_path):
    """Loads and reads env file

    Args:
        file_path (str): file name for the env file
    """
    with open(file_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value


# Load environment variables from .env file
load_env_file('train.env')


# data loader
# create Dataset
class CustomImageDataset(Dataset):
    """Create a custom Dataset for the pytorch dataloader

    Args:
        Dataset (class): torch.utils.data Dataset class
    """
    def __init__(self, img_dir, transform=None):
        """Initializes the CustomImageDataset class

        Args:
            img_dir (str): _description_
            transform (torch transform, optional): Is an instance of a torch transformer. Defaults to None.
        """
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """Return the length of the dataset

        Returns:
            int: Number of files in the dataset
        """
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        """Generator instance

        Args:
            idx (int): Index of the next object from the datatset

        Returns:
            np.array: A np array from an image scaled to interval [0, 1]
        """
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Scale the tensor to the interval [0, 1]
        image_tensor = image.float() / 255.0

        return image_tensor


def main():
    """This function runs the train loop for the VIS AI
    """
    # load parameter from env
    images_folder = str(os.getenv("IMAGES_FOLDER"))

    LR = float(os.getenv("LR"))
    PATIENCE = int(os.getenv("PATIENCE"))
    IMAGE_SIZE = ast.literal_eval(os.getenv("IMAGE_SIZE"))
    CHANNELS = int(os.getenv("CHANNELS"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))
    EPOCHS = int(os.getenv("EPOCHS"))

    smaller = 4 * 4 * 2
    SHAPE_BEFORE_FLATTENING = (128, IMAGE_SIZE[0] // smaller, IMAGE_SIZE[1] // smaller)
    num_workers = 0

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"The training uses this device: {DEVICE}")

    # create directories
    output_dir = './models'
    os.makedirs(output_dir, exist_ok=True)

    # define the transformation to be applied to the data
    transform = transforms.Compose([v2.PILToTensor()])

    dataset = CustomImageDataset(images_folder, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    # instantiate the encoder and decoder models
    encoder = Encoder(IMAGE_SIZE, EMBEDDING_DIM).to(DEVICE)
    decoder = Decoder(EMBEDDING_DIM, SHAPE_BEFORE_FLATTENING).to(DEVICE)
    # pass the encoder and decoder to VAE class
    vae = VAE(encoder, decoder)

    # instantiate optimizer and scheduler
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=PATIENCE)

    # initialize the best validation loss as infinity
    best_val_loss = float("inf")

    # Model start time for path names
    model_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # define model_weights, reconstruction & real before training images paths
    best_model_name = f'best_vae_{model_start_time}_embedding-dim_{EMBEDDING_DIM}.pt'.replace(':', '-').replace(' ', '_')
    MODEL_WEIGHTS_PATH = os.path.join(output_dir, best_model_name)

    # start training by looping over the number of epochs
    for epoch in range(EPOCHS):
        logging.info(f"Epoch {epoch + 1}")
        # set the vae model to train mode
        # and move it to CPU/GPU
        vae.train()
        vae.to(DEVICE)
        running_loss = 0.0
        # loop over the batches of the training dataset
        for batch_idx, data in enumerate(dataloader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            # forward pass through the VAE
            pred = vae(data)
            # compute the VAE loss
            loss, recon_loss, kld_loss = vae_loss(pred, data)
            # backward pass and optimizer step
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # compute average loss for the epoch
        train_loss = running_loss / len(dataloader)
        # print loss at every epoch
        logging.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Recon Loss: {recon_loss:.4f} | KL Loss: {kld_loss:.4f}")
        # save best vae model weights based on validation loss
        if recon_loss < best_val_loss:
            best_val_loss = recon_loss
            torch.save({"vae": vae.state_dict()}, MODEL_WEIGHTS_PATH,)
        # adjust learning rate based on the validation loss
        scheduler.step(train_loss)
        
        logging.info("Training done")

    model_name = f'torch_model_big_boat_mnist_sea_{model_start_time}_epochs_{EPOCHS}_batchsize_{BATCH_SIZE}_embedding-dim_{EMBEDDING_DIM}.pt'.replace(':', '-').replace(' ', '_')
    model_name = os.path.join(output_dir, model_name)

    torch.save(vae.state_dict(), model_name)
    logging.info(f"Latest model saved at: {model_name}, with loss: {recon_loss}")
    logging.info(f"Best model saved at: {best_model_name}, with loss: {best_val_loss}")

 
if __name__=="__main__":
    main()
