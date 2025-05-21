# libraries
import logging
import numpy as np
from sklearn.cluster import HDBSCAN # type: ignore
from scipy.ndimage import maximum_filter

import torch

from vae_model import Decoder, Encoder, VAE, model_prediction

# inference class
class VISAIModel:
    """Wrapper class for VIS AI Model, Variational Autoencoder 
    """
    def __init__(self, model_name: str, fullRes: int, num_of_lines: int) -> None:
        """Initialize the class

        Args:
            model_name (str): _description_
            fullRes (int): _description_
            num_of_lines (int): _description_
        """
        # model parameters
        self.model_name = model_name
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.IMAGE_SIZE = (fullRes, num_of_lines, 3)
        self.EMBEDDING_DIM = int(model_name.split(".")[-2].split("_")[-1])
        print(self.EMBEDDING_DIM)
        smaller = 4 * 4 * 2
        self.Threshold_grayscale = 144 # 256 * (19/32) = 144
        self.SHAPE_BEFORE_FLATTENING = (128, self.IMAGE_SIZE[0] // smaller, self.IMAGE_SIZE[1] // smaller)

        self._load_model()
        
    def _load_model(self) -> None:
        """Load a variational autoencoder (vae) from a pytorch state model (.pt.pth)
        """
        # instantiate the encoder and decoder models
        encoder = Encoder(self.IMAGE_SIZE, self.EMBEDDING_DIM).to(self.DEVICE)
        decoder = Decoder(self.EMBEDDING_DIM, self.SHAPE_BEFORE_FLATTENING).to(self.DEVICE)

        self.vae_model = VAE(encoder, decoder).to(self.DEVICE)
        self.vae_model.load_state_dict(torch.load(self.model_name, map_location=torch.device(self.DEVICE)))
        self.vae_model.eval()

    def inference(self, val_img: np.ndarray, alpha: np.ndarray) -> tuple[int, np.ndarray]:
        """Runs the VAE model and filters result with alpha channel

        Args:
            val_img (np.array): RGB image for the anomaly detection
            alpha (np.array): Alpha channel as flag. Useable lines are flagged with ones (1), else (0)

        Returns:
            tuple[int, list]: size of list, list of hotspots
        """
        data = model_prediction(self.vae_model, val_img, self.Threshold_grayscale, self.IMAGE_SIZE,  self.DEVICE)

        bigger = (self.Threshold_grayscale < data).astype(int) # binary mask
        data = bigger * data

        alpha = np.squeeze(alpha)
        data = data * alpha
        
        self.data = data

        if np.max(data) <= 0:
            return 0, np.array([])
        else:
            # get hotspots
            # set window size
            window_size = 4

            # create mask and find max values in a window
            arr_new = data*(data == maximum_filter(data,footprint=np.ones((window_size, window_size))))

            # Scale to interval [0, 1] w/ float values with two decimals
            arr_new = np.round(arr_new/np.max(arr_new), 2)

            non_zeros = np.nonzero(arr_new)
            hotspots = np.array([(x, y) for x, y in zip(non_zeros[1], non_zeros[0])])

            weighted_hotspots = np.array([[arr_new[y][x], x, y] for x, y in zip(non_zeros[1], non_zeros[0])])
            weighted_hotspots = weighted_hotspots[weighted_hotspots[:, 0].argsort()[::-1]]
            select_subset = 2**8
            hotspots = weighted_hotspots[:select_subset,1:] # select subset of pixels, select only pixel information

            # Unsupervised Clustering to get number of classes
            hdb = HDBSCAN(min_cluster_size=5, store_centers='medoid',n_jobs=-1).fit(hotspots)

            try:
                weighted_hotspots = np.array([[arr_new[int(y)][int(x)], x, y] for x, y in hdb.medoids_])
            except Exception as e:
                logging.error("Exception occurred", exc_info=True)

            return len(hdb.medoids_), np.array(weighted_hotspots)


if __name__=='__main__':
    pass
