import json
import asyncio
from enum import Enum
from urllib.parse import urljoin

import geopandas as gpd
from shapely.geometry import box
import pandas as pd

from ultralytics import YOLO
from fastapi import Request, Response

from ai_inference.inference.common import ModelDescription, ModelInference, PredictInput
from ai_inference.inference.routes import router
from ai_inference.job import Job


model_meta = ModelDescription(
    name="yolov8",
    description="""\
!Important Info: For research purposes only!
This model is trained to detect 4 different classes: 1_Plastic, 7_Metal, 300_Others, 5_Paper
More informations and the pretrained YOLO models can be found here: <https://huggingface.co/Ultralytics/YOLOv8>
We used a pretrained YOLOv8x model.
""",
)


class OutputType(str, Enum):
    DOWNLOAD = "Download"
    RESPONSE = "Response"


class PLD_PredictInput(PredictInput):
    pass


class YOLOv8(ModelInference):
    model = None
    
    @staticmethod
    def description():
        return model_meta

    def __init__(self, job: Job | None):
        super().__init__(job)
        
    @classmethod
    def warm_up(
        cls,
        conf_threshold=0.5,
        model_path="./models/best_yolov8x_4_classes.pt"
    ):
        cls.model = YOLO(model_path)
        cls.conf_threshold = conf_threshold
        
        with open("ai_inference/inference/maskrcnn_4classes/categories.json") as f:
            data = json.load(f)
        
        cls.categories = [d["name"] for d in data["categories"]]
        
    def _do_predict(self, input_file: str, input_parameters: PredictInput) -> gpd.GeoDataFrame:
        results = YOLOv8.model(input_file)
        
        results = results[0]
        
        boxes = results.boxes

        # Filter boxes, labels, categories, and scores based on the threshold
        geometries = []
        labels = []
        cats = []
        scores = []

        for i, score in enumerate(boxes.conf):
            if score >= self.conf_threshold:
                xmin, ymin, xmax, ymax = boxes.xyxy[i]
                geometries.append(box(xmin, ymin, xmax, ymax))
                labels.append(boxes.cls[i])
                cats.append(self.categories[int(boxes.cls[i])])
                scores.append(score)

        
        # geometries = [box(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in boxes.xyxy]
        
        # labels = [l for l in boxes.cls]
        # cats = [self.categories[int(l)] for l in boxes.cls]
        
        # scores = [s for s in boxes.conf]

        data = {'geometry': geometries, 'label': labels, "categories": cats, 'score': scores}

        gdf = gpd.GeoDataFrame(pd.DataFrame(data))

        return gdf
        

@router.post(f"/{model_meta.name}/", status_code=201)
async def call_predict(input: PLD_PredictInput, request: Request, response: Response) -> dict:
    from ai_inference.main import add_new_job_to_queue

    job: Job = add_new_job_to_queue(request)

    inference = YOLOv8(job)
    asyncio.create_task(inference.predict(input))

    response.headers["location"] = urljoin(str(request.base_url), f"jobs/{job.job_id}")
    return {"job_id": job.job_id, "status": job.status, "msg": job.msg}
 
    
if __name__ == '__main__':
    # python -m ai_inference.inference.yolov8.inference
    from pyproj import CRS as PyprojCRS
    import matplotlib.pyplot as plt
    from PIL import Image

    job = None
    input_file = "ai_inference/test_images/Plot_4_.png"
    parameters = ""
    inference = YOLOv8(job)
    inference.warm_up()
    result_gdf = inference._do_predict(input_file, parameters)
    epsg_code = 4326
    crs = PyprojCRS.from_epsg(epsg_code)
    result_gdf = result_gdf.set_crs(crs)
    result_gdf.to_file("Shape_files/yolo_bboxes_with_metadata.shp")

    # Replace 'your_shapefile.shp' with the path to your shapefile
    gdf = gpd.read_file("Shape_files/yolo_bboxes_with_metadata.shp")

    # Display the first few rows of the GeoDataFrame
    print(gdf.head())

    # Get basic information about the GeoDataFrame
    print(gdf.info())

    image = Image.open(input_file)
    # Plot the geometries
    gdf.plot()
    plt.imshow(image)
    plt.show()

    print(YOLOv8.categories)
