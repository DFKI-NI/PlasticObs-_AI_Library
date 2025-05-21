import os
import re
import json
import base64
import asyncio

from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import ParseResult, parse_qs, urlparse

import requests
import torch
from loguru import logger
from pydantic import BaseModel
from pyproj import CRS as PyprojCRS
import geopandas as gpd
from shapely.geometry import box

from ai_inference.job import Job, TaskStatus
from ai_inference.utils import generate_random_filename, generate_output_filename


class PredictInput(BaseModel):
    pk: str
    title: str
    imageUrl: str
    inferenceGroup: str = None


class ModelDescription(BaseModel):
    name: str
    description: Optional[str] = ""


class ModelInference(ABC):

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, job: Job | None):
        self.job = job if job else Job(None)
        self.ssl_verify = os.getenv("SSL_VERIFY", "False") == "True"
        # self.geonode_base_url: ParseResult | None
        # self.parsed_qs = dict[str, list[str]] | None

    @staticmethod
    def warm_up():
        pass

    @staticmethod
    @abstractmethod
    def description() -> ModelDescription:
        pass

    @abstractmethod
    def _do_predict(self, input_file: str, input_parameters: PredictInput) -> gpd.GeoDataFrame:
        pass

    def _prepare_auth(self):
        headers = {}
        if "access_token" in self.parsed_qs:
            access_token = self.parsed_qs["access_token"][0]
            token_param = f"access_token={access_token}"
        else:
            encoded = base64.b64encode(b"admin:admin")
            credentials = encoded.decode()
            headers["Authorization"] = f"Basic {credentials}"
        return headers, token_param

    async def _wait_until_done(self, exec_id, sleep_seconds=1):
        base_url = self.geonode_base_url.geturl()
        headers, access_token = self._prepare_auth()
        execution_url = f"{base_url}/api/v2/executionrequest/{exec_id}?{access_token}"

        async def check_status():
            response = requests.get(execution_url, headers=headers).json()
            return response["request"]["status"], response["request"]["output_params"]

        while True:
            await asyncio.sleep(sleep_seconds)
            status, output_params = await check_status()
            if status == "finished":
                # we expect just one uploaded resource here
                return output_params["resources"][0]["id"]
            if status == "failed":
                raise Exception("\n".join(output_params["errors"]))

    async def predict(self, input_parameters: PredictInput):
        logger.info(f"Starting inference for job {str(self.job)}")
        try:
            group = input_parameters.inferenceGroup
            download_url = input_parameters.imageUrl
            logger.debug(f"Download image from '{download_url}' ...")
            input_file = self._download_from_wms(download_url)

            required_parameters = ["bbox", "access_token"]
            for p in required_parameters:
                if self.parsed_qs and p not in self.parsed_qs:
                    raise Exception(f"imageUrl is missing '{p}' query parameter ")

            logger.debug("Start prediction ...")
            gdf = self._do_predict(input_file, input_parameters)

            model_name = self.description().name
            file_name = input_parameters.title.split(".")[0]
            output_file = generate_output_filename(file_name, model_name, "shp")
            logger.debug(f"Creating Shapefile '{output_file}' ...")
            self.save_gdf_with_crs(gdf, output_file)

            logger.debug(f"Upload result '{output_file}' to GeoNode ...")
            upload_response = self.upload_to_geonode(output_file, group)
            exec_status_response = json.loads(upload_response.decode())
            exec_id = exec_status_response["execution_id"]
            timeout = 60 * 60  # 1 hour
            uploaded_pk = None
            try:
                uploaded_pk = await asyncio.wait_for(self._wait_until_done(exec_id=exec_id), timeout=timeout)
            except asyncio.TimeoutError:
                raise Exception("Timeout while uploading result. Cannot link datasets.")

            base_url = self.geonode_base_url.geturl()
            url = f"{base_url}/api/v2/resources/{uploaded_pk}/linked_resources"
            headers, token_param = self._prepare_auth()
            headers.update({"Content-Type": "application/json"})
            response = requests.request(
                "POST",
                f"{url}?{token_param}",
                headers=headers,
                json={"target": [input_parameters.pk]},
                timeout=10,
                verify=self.ssl_verify,
            )

            if response is not None and response.status_code > 400:
                logger.error(response.json())
                raise Exception("Could not link uploaded result to input dataset!")

            self.job.status = TaskStatus.COMPLETE
        except Exception as e:
            self.job.status = TaskStatus.FAILED
            self.job.msg = f"{e}"
            logger.exception(e)

    def _download_from_wms(self, url: str) -> str:
        """Takes given URL to download and to save image locally.

        Args:
            url (str): The URL to download the image from.

        Raises:
            HTTPException: in case download response has http status != 200

        Returns:
            str: the local file name of the downloaed image.
        """
        parsed_url = url if isinstance(url, ParseResult) else urlparse(url)
        if parsed_url.scheme == "file" or parsed_url.netloc is None:
            raise Exception("Inference on local files are not supported!")
        default_base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        geonode_base_url = urlparse(os.getenv("GEONODE_BASE_URL", default_base_url))
        geoserver_base_url = urlparse(os.getenv("GEOSERVER_BASE_URL", geonode_base_url.geturl()))
        parsed_url = parsed_url._replace(
            scheme=geoserver_base_url.scheme, netloc=geoserver_base_url.netloc, path="/gs/ows"
        )

        try:
            response = requests.get(parsed_url.geturl(), verify=self.ssl_verify, timeout=60)
            if response:
                logger.debug(f"Download image response status: {response.status_code}")
        except Exception as e:
            msg = f"Could not download image from {parsed_url.geturl()}"
            logger.error(msg, e)
            raise Exception(msg)

        self.geonode_base_url = geonode_base_url

        def parse_qs_with_casesensitives(query: str, case_sensitives=[]):
            """access_token is case sensitive"""
            parsed = parse_qs(query.lower())
            for cs_param in case_sensitives:
                pattern = f'{cs_param}=([^&]+)'
                match = re.search(pattern, query)
                if match:
                    parsed[cs_param.lower()] = [match.group(1)]
            return parsed

        case_sensitive_params = ["access_token"]
        self.parsed_qs = parse_qs_with_casesensitives(parsed_url.query, case_sensitive_params)

        local_filename = generate_random_filename("png")
        if response.status_code == 200:
            # Save the image data to a local file
            with open(local_filename, "wb") as f:
                f.write(response.content)

            if "text/xml" in response.headers["Content-Type"]:
                message = "Downloading from WMS raised an error"
                with open(local_filename, "r") as report:
                    details = "".join(report.readlines())
                    raise Exception(f"{message}: {details}")

            logger.debug(f"Image downloaded and saved as {local_filename}")
        else:
            logger.error(f"Error downloading image [{response.status_code}]")
            raise Exception("Failed to get image. Check logs for details.")

        return local_filename

    def upload_to_geonode(self, file_path: str, group: str = None):
        """Uploads image to a GeoNode instance.

        Args:
            file_path (str, Path): file path to a GeoTiff to be uploaded to GeoNode
        """
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        base_path_filename = os.path.splitext(file_path)[0]
        files = [
            (
                "base_file",
                (
                    os.path.basename(f"{base_filename}.shp"),
                    open(f"{base_path_filename}.shp", "rb"),
                    "application/octet-stream",
                ),
            ),
            (
                "dbf_file",
                (
                    os.path.basename(f"{base_filename}.dbf"),
                    open(f"{base_path_filename}.dbf", "rb"),
                    "application/octet-stream",
                ),
            ),
            (
                "shx_file",
                (
                    os.path.basename(f"{base_filename}.shx"),
                    open(f"{base_path_filename}.shx", "rb"),
                    "application/octet-stream",
                ),
            ),
            (
                "cpg_file",
                (
                    os.path.basename(f"{base_filename}.cpg"),
                    open(f"{base_path_filename}.cpg", "rb"),
                    "application/octet-stream",
                ),
            ),
            (
                "prj_file",
                (
                    os.path.basename(f"{base_filename}.prj"),
                    open(f"{base_path_filename}.prj", "rb"),
                    "application/octet-stream",
                ),
            ),
        ]

        # Ensure all files are opened correctly
        # for file_type, (filename, file_obj, mime_type) in files:
        #     if not file_obj:
        #         raise Exception(f"Could not open file: {filename}")

        upload_base_url = self.geonode_base_url.geturl()
        url = f"{upload_base_url}/api/v2/uploads/upload"
        headers, token_param = self._prepare_auth()

        response = requests.request(
            "POST",
            f"{url}?{token_param}",
            headers=headers,
            files=files,
            data={
                "action": "upload",
                "overwrite_existing_layer": "false",
                "custom": json.dumps(
                    {"group": group, "title": base_filename, "abstract": "PlasticObs+ Inference Result."}
                ),
            },
            timeout=30,
            verify=self.ssl_verify,
        )

        logger.debug("Upload done!")
        if not response:
            error = response.json()
            logger.error(error)
            raise Exception(f"Could not upload result to GeoNode: {error}")
        else:
            return response.content

    # def generate_raster_for_image(self, image: np.ndarray, out_file_name: str, epsg_code: int = 4326):
    #     if "srs" in self.parsed_qs:
    #         epsg_code = self.parsed_qs["srs"][0].split(":")[-1] or epsg_code
    #     ll_x, ll_y, ur_x, ur_y = self.parsed_qs["bbox"][0].split(",")

    #     # Convert the array to match rasterio standard (shape: (3, :, :))
    #     image = np.moveaxis(image, -1, 0)
    #     channels, height, width = image.shape  # rasterio style

    #     # Calculate resolution (pixel size)
    #     pixel_size_x = (float(ur_x) - float(ll_x)) / width  # num_columns
    #     pixel_size_y = (float(ur_y) - float(ll_y)) / height  # num_rows

    #     # write the image data
    #     out_meta = {
    #         "driver": "GTiff",
    #         "dtype": rasterio.uint8,
    #         "height": height,
    #         "width": width,
    #         "transform": from_origin(float(ll_x), float(ur_y), pixel_size_x, pixel_size_y),
    #         "crs": RasterioCRS.from_epsg(epsg_code),
    #         "count": channels,  # RGB images
    #     }

    #     # write final image
    #     with rasterio.open(out_file_name, "w", **out_meta) as dest:
    #         dest.write(image)

    def save_gdf_with_crs(self, gdf: gpd.GeoDataFrame, file_path: str):
        """
        Adds the specified EPSG code as CRS to the GeoDataFrame and writes it to a shapefile.

        Parameters:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to which the CRS will be added.
        file_path (str): The path where the shapefile will be saved.
        """
        #epsg_code = self.parsed_qs["crs"][0].split(":")[-1]
        if "crs" in self.parsed_qs:
            epsg_code = self.parsed_qs["crs"][0].split(":")[-1]
        elif "srs" in self.parsed_qs: # enable batch processing
            epsg_code = self.parsed_qs["srs"][0].split(":")[-1]

        ll_x, ll_y, ur_x, ur_y = map(float, self.parsed_qs["bbox"][0].split(","))

        image_width = int(self.parsed_qs["width"][0])
        image_height = int(self.parsed_qs["height"][0])

        # FIXME: Overwrite image size
        # image_width, image_height = 1000, 1000

        # Calculate resolution (pixel size)
        pixel_size_x = (ur_x - ll_x) / image_width
        pixel_size_y = (ur_y - ll_y) / image_height

        # Transform pixel coordinates to geographic coordinates
        def transform_pixel_to_geo(px_xmin, px_ymin, px_xmax, px_ymax):
            geo_xmin = ll_x + (px_xmin * pixel_size_x)
            geo_ymin = ll_y - (px_ymin * pixel_size_y) + (ur_y - ll_y)
            geo_xmax = ll_x + (px_xmax * pixel_size_x)
            geo_ymax = ll_y - (px_ymax * pixel_size_y) + (ur_y - ll_y)
            return box(geo_xmin, geo_ymin, geo_xmax, geo_ymax)

        gdf['geometry'] = gdf['geometry'].apply(lambda geom: transform_pixel_to_geo(*geom.bounds))

        crs = PyprojCRS.from_epsg(epsg_code)
        gdf = gdf.set_crs(crs)

        gdf.to_file(file_path)


if __name__ == "__main__":
    pass
