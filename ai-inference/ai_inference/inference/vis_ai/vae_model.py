# libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from PIL import Image
from torch.distributions.normal import Normal


class LogCoshLoss(nn.Module):
    """Custom Loss class

    Args:
        nn (module): Torch neural network module
    """

    def __init__(self):
        """Initialize Custom Loss class"""
        super().__init__()

    def forward(self, y_t, y_prime_t):
        """Calculate forward pass of the loss function

        Args:
            y_t (image): reconstructed image
            y_prime_t (image): original image

        Returns:
            float: Loss over the variance between the images
        """
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


def vae_gaussian_kl_loss(mu, logvar):
    """Calculate the kl loss of thew autoencoder

    Args:
        mu (float): one of the return values from the autoencoder the prediction
        logvar (float): one of the return values from the autoencoder the prediction

    Returns:
        float: kl loss of the autoencoder
    """
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return KLD.mean()


def reconstruction_loss(x_reconstructed, x):
    """Calculate the reconstruction loss

    Args:
        x_reconstructed (image): reconstructed image
        x (image): original image

    Returns:
        float: Loss over the variance between the images
    """
    bce_loss = LogCoshLoss()  # nn.BCELoss()
    return bce_loss(x_reconstructed, x)


def vae_loss(y_pred, y_true):
    """Calculate the losses for the VAE

    Args:
        y_pred (image): reconstructed image
        y_true (image): original image

    Returns:
        float: total loss, reconstruction loss and kl loss
    """
    mu, logvar, recon_x = y_pred
    recon_loss = reconstruction_loss(recon_x, y_true)
    kld_loss = vae_gaussian_kl_loss(mu, logvar)
    return (500 * recon_loss + kld_loss), 500 * recon_loss, kld_loss


# define a class for sampling
class Sampling(nn.Module):
    """This class will be used in the encoder for sampling in the latent space

    Args:
        nn (module): Torch neural network module
    """

    def forward(self, z_mean, z_log_var):
        """Forward pass for the Sampling class

        Args:
            z_mean (array): latent space dense layer results
            z_log_var (array): latent space dense layer results

        Returns:
            list: latent space
        """
        # get the shape of the tensor for the mean and log variance
        batch, dim = z_mean.shape
        # generate a normal random tensor (epsilon) with the same shape as z_mean
        # this tensor will be used for reparameterization trick
        epsilon = Normal(0, 1).sample((batch, dim)).to(z_mean.device)
        # apply the reparameterization trick to generate the samples in the
        # latent space
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


# define the encoder
class Encoder(nn.Module):
    """The encoder for the VAE

    Args:
        nn (module): Torch neural network module
    """

    def __init__(self, image_size, embedding_dim):
        """Initializes the encoder class

        Args:
            image_size (array): Shape of the images
            embedding_dim (int): Size of the latent space
        """
        super(Encoder, self).__init__()
        # define the convolutional layers for downsampling and feature
        # extraction
        self.conv1 = nn.Conv2d(3, 32, 3, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=4, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        # stride1 * stride2 * stride3
        smaller = 4 * 4 * 2
        # define a flatten layer to flatten the tensor before feeding it into
        # the fully connected layer
        self.flatten = nn.Flatten()
        # define fully connected layers to transform the tensor into the desired
        # embedding dimensions
        self.fc_mean = nn.Linear(128 * (image_size[1] // smaller) * (image_size[0] // smaller), embedding_dim)
        self.fc_log_var = nn.Linear(128 * (image_size[1] // smaller) * (image_size[0] // smaller), embedding_dim)
        # initialize the sampling layer
        self.sampling = Sampling()

    def forward(self, x):
        """Forward pass for the VAE

        Args:
            x (image): Input image

        Returns:
            array: latent space variables
        """
        # apply convolutional layers with relu activation function
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # flatten the tensor
        x = self.flatten(x)

        x = x.float()
        # print(x.shape)
        # get the mean and log variance of the latent space distribution
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        # sample a latent vector using the reparameterization trick
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


# define the decoder
class Decoder(nn.Module):
    """The decoder class for an AutoEncoder

    Args:
        nn (module): Torch neural network module
    """

    def __init__(self, embedding_dim, shape_before_flattening):
        """Initializes the decoder class

        Args:
            embedding_dim (int): Size of the latent space
            shape_before_flattening (array): shape of image before flattening
        """
        super(Decoder, self).__init__()
        # define a fully connected layer to transform the latent vector back to
        # the shape before flattening
        self.fc = nn.Linear(
            embedding_dim,
            shape_before_flattening[0] * shape_before_flattening[1] * shape_before_flattening[2],
        )
        # define a reshape function to reshape the tensor back to its original
        # shape
        self.reshape = lambda x: x.view(-1, *shape_before_flattening)
        # define the transposed convolutional layers for the decoder to upsample
        # and generate the reconstructed image
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=4, padding=1, output_padding=3)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 3, stride=4, padding=1, output_padding=3)

    def forward(self, x):
        """Forward pass for the VAE

        Args:
            x (list): latent space from the encoder

        Returns:
            array: reconstructed image
        """
        # pass the latent vector through the fully connected layer
        x = self.fc(x)
        # reshape the tensor
        x = self.reshape(x)
        # apply transposed convolutional layers with relu activation function
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        # apply the final transposed convolutional layer with a sigmoid
        # activation to generate the final output
        x = torch.sigmoid(self.deconv3(x))
        return x


# define the vae class
class VAE(nn.Module):
    """VAE Model

    Args:
        nn (module): Torch neural network modul
    """

    def __init__(self, encoder, decoder):
        """Initialize the VAE Model class

        Args:
            encoder (class): The encoder for the VAE
            decoder (class): The decoder class for an AutoEncoder
        """
        super(VAE, self).__init__()
        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """Forward pass for the VAE

        Args:
            x (image): Input image

        Returns:
            array: reconstructed image
        """
        # pass the input through the encoder to get the latent vector
        z_mean, z_log_var, z = self.encoder(x)
        # pass the latent vector through the decoder to get the reconstructed
        # image
        reconstruction = self.decoder(z)
        # return the mean, log variance and the reconstructed image
        return z_mean, z_log_var, reconstruction


def predict_single(model, test_image, DEVICE):
    """Predict one image w/ a VAE Model

    Args:
        model (class): Trained torch VAE model
        test_image (array): Image for the prediction
        DEVICE (str): torch.device name

    Returns:
        array: reconstructed image
    """
    img = test_image
    img = np.expand_dims(img, axis=0)
    img = (
        torch.Tensor(img.reshape(-1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
        .permute(0, 3, 1, 2)
        .to(DEVICE)
    )

    z_mean, z_log_var, z = model.encoder(img)

    # SAMPLE IMAGES
    with torch.no_grad():
        pred = model.decoder(z.to(DEVICE)).detach().cpu().numpy()

    return pred


# encoding and decoding with a given model on one (1) image
def model_prediction(vae_model, image, threshold, img_shape, DEVICE):
    """Make model prediction and anomaly filtering

    Args:
        vae_model (class): Trained torch VAE model
        image_name (np.array): rgb image (uint8)
        threshold (int): Lower threshold for the anomaly detection
        img_shape (list): Shape of the image
        DEVICE (str): torch.device name

    Returns:
        array: negative image for the anomaly detection
    """
    # open image and norm to float \[0,1]
    image_array_plastic = np.array(image) / 255

    image_array = predict_single(vae_model, image_array_plastic, DEVICE)
    images_array = np.array(image_array)
    images_array = images_array.transpose((0, 2, 3, 1))
    images_array = images_array.reshape(img_shape)
    im = Image.fromarray((images_array * 255).astype(np.uint8))

    image = Image.fromarray(np.array(image).astype(np.uint8))

    # get gray scaled images
    images_array_plastic = np.array(image.convert('L'))
    images_array_gen = np.array(im.convert('L'))

    # get differences in the image
    img_substract = abs(images_array_plastic.astype(np.float32) - images_array_gen.astype(np.float32))
    # filter out small/random noises (waves, etc.)
    bigger = (threshold < img_substract).astype(int)  # binary mask
    data = bigger * img_substract

    return data


if __name__ == '__main__':
    pass
