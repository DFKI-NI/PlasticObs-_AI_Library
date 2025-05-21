import os
import asyncio
from enum import Enum
from urllib.parse import urljoin

import numpy as np
import torch
from fastapi import Request, Response
from loguru import logger
from PIL import Image
import geopandas as gpd
from shapely.geometry import Point

from ai_inference.inference.common import ModelDescription, ModelInference, PredictInput
from ai_inference.inference.routes import router
from ai_inference.inference.vis_ai.vae_model import VAE, Decoder, Encoder, model_prediction
from ai_inference.job import Job, JobStatus
from ai_inference.utils import slice_image, stitch_image

model_meta = ModelDescription(
    name="vis_ai",
    description="""\
        The VAE model is designed to learn a compressed representation of the input data
        and subsequently reconstruct the images from this representation. This approach
        is employed for the generation of new, similar images for the purpose of anomaly
        detection on plastic waste in the ocean and coastal areas.\
        You can find the original VAE paper here: <doi: 10.48550/ARXIV.1312.6114.>\
    """,
)


class OutputType(str, Enum):
    DOWNLOAD = "Download"
    RESPONSE = "Response"


class VisAIInference(ModelInference):
    model = None
    image_size = (3200, 480, 3)
    threshold_grayscale = 192

    @staticmethod
    def description():
        return model_meta

    def __init__(self, job: Job | None):
        super().__init__(job)

    @classmethod
    def warm_up(
        cls,
        model_name="torch_model_optimare_2024-03-21_15-58-01_epochs_10_batchsize_32_EmbeddingDim_35.pt",
        full_resolution=3200,
        num_of_lines=480,
        threshold_grayscale=144,
    ):
        image_size = (full_resolution, num_of_lines, 3)
        smaller = 4 * 4 * 2
        embedding_dim = int(model_name.split(".")[-2].split("_")[-1])
        shape_before_flattening = (128, image_size[0] // smaller, image_size[1] // smaller)
        encoder = Encoder(image_size, embedding_dim).to(cls.DEVICE)
        decoder = Decoder(embedding_dim, shape_before_flattening).to(cls.DEVICE)

        model = VAE(encoder, decoder)
        path = os.path.join(os.getcwd(), "models", model_name)
        model.load_state_dict(torch.load(path, map_location=cls.DEVICE, weights_only=True))
        cls.model = model.eval()
        cls.image_size = image_size
        cls.threshold_grayscale = threshold_grayscale

    def _do_predict(self, input_file: str, input_parameters: PredictInput) -> gpd.GeoDataFrame:
        with Image.open(input_file).convert("RGB") as input_image:
            logger.debug("Reading input image as numpy array ...")

            image_size = self.image_size
            input_image = np.array(input_image)
            slices, pad_height, pad_width = slice_image(input_image, image_size[0], image_size[1])

            results = []
            for slice in slices:
                if "alpha" in input_parameters:

                    # TODO provisioning by request?
                    alpha = input_parameters.alpha
                else:
                    alpha = np.ones((image_size[0], image_size[1]))
                # alpha = alpha.resize(self.IMAGE_SIZE[0], self.IMAGE_SIZE[1])

                if slice.shape == image_size:
                    model = VisAIInference.model
                    device = VisAIInference.DEVICE
                    threshold = VisAIInference.threshold_grayscale
                    data = model_prediction(model, slice, threshold, image_size, device)
                    bigger = (threshold < data).astype(int)  # binary mask
                    data = bigger * data

                    data = alpha * data

                else:
                    data = np.zeros((slice.shape[0], slice.shape[1]))

                results.append(data)

            # TODO what to do with the medoid outputs
            """
            if np.max(data) <= 0:
                pass  # do nothing
            else:
                # get hotspots
                # set window size
                window_size = 4

                # create mask and find max values in a window
                arr_new = data * (data == maximum_filter(data, footprint=np.ones((window_size, window_size))))

                # Scale to interval [0, 1] w/ float values with two decimals
                arr_new = np.round(arr_new / np.max(arr_new), 2)

                non_zeros = np.nonzero(arr_new)
                hotspots = np.array([(x, y) for x, y in zip(non_zeros[1], non_zeros[0])])

                weighted_hotspots = np.array([[arr_new[y][x], x, y] for x, y in zip(non_zeros[1], non_zeros[0])])
                weighted_hotspots = weighted_hotspots[weighted_hotspots[:, 0].argsort()[::-1]]
                hotspots = weighted_hotspots[:, 1:]

                # Unsupervised Clustering to get number of classes
                hdb = HDBSCAN(min_cluster_size=5, store_centers='medoid', n_jobs=-1).fit(hotspots)  ## noqa
                """

            stitched_image = stitch_image(
                results, input_image.shape[0], input_image.shape[1], image_size[0], image_size[1], pad_height, pad_width
            )
            with Image.fromarray(stitched_image.astype(np.uint8), mode='RGB') as image_pil:
                grayscale_image_pil = image_pil.convert('L')
                anomaly_array = np.array(grayscale_image_pil)

            # List to store geometries and scores
            geometries = []
            scores = []

            # Iterate over the array to find anomalies
            for i in range(anomaly_array.shape[0]):
                for j in range(anomaly_array.shape[1]):
                    if anomaly_array[i, j] > threshold:
                        # Create a point geometry for each anomaly
                        geometries.append(Point(j, i))
                        scores.append(anomaly_array[i, j])

            # Create a GeoDataFrame
            data = {'geometry': geometries, 'score': scores}
            gdf = gpd.GeoDataFrame(data)

            return gdf


@router.post(f"/{model_meta.name}/", status_code=201)
async def call_predict(input: PredictInput, request: Request, response: Response) -> JobStatus:
    from ai_inference.main import add_new_job_to_queue

    job: Job = add_new_job_to_queue(request)

    inference = VisAIInference(job)
    asyncio.create_task(inference.predict(input))

    response.headers["location"] = urljoin(str(request.base_url), f"jobs/{job.job_id}")
    return {"job_id": job.job_id, "status": job.status, "msg": job.msg}


if __name__ == '__main__':
    from pyproj import CRS as PyprojCRS

    job = None
    input_file = "ai_inference/test_images/20240625_Hur_20240625_001_HRVIS_56_1.png"
    parameters = ""
    inference = VisAIInference(job)
    inference.warm_up()
    result_gdf = inference._do_predict(input_file, parameters)
    epsg_code = 4326
    crs = PyprojCRS.from_epsg(epsg_code)
    result_gdf = result_gdf.set_crs(crs)
    result_gdf.to_file("Shape_files/vis_ai_bboxes_with_metadata.shp")
