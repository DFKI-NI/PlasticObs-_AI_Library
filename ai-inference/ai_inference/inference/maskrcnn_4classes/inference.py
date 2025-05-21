import asyncio
import json
from enum import Enum
from urllib.parse import urljoin

import geopandas as gpd
from shapely.geometry import box
import pandas as pd

import torch
from fastapi import Request, Response
from PIL import Image
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

from ai_inference.inference.common import ModelDescription, ModelInference, PredictInput
from ai_inference.inference.maskrcnn_resnet50 import preprocessing
from ai_inference.inference.routes import router
from ai_inference.job import Job

model_meta = ModelDescription(
    name="maskrcnn_4classes",
    description="""\
This model only has 4 classes.
We've repurposed the Mask R-CNN model, originally designed for object instance segmentation, for plastic waste detection using transfer learning.
By fine-tuning a pre-trained Mask R-CNN on a custom dataset of plastic waste images, the model now efficiently detects and segments plastic waste, particularly bottles.
This generates high-quality segmentation masks, facilitating the automatic identification and separation of plastic waste for recycling.
The approach has shown promising results, indicating its potential for practical applications in environmental conservation.\
You can find the original Mask R-CNN model paper here: <https://arxiv.org/abs/1703.06870>_\
""",
)

class OutputType(str, Enum):
    DOWNLOAD = "Download"
    RESPONSE = "Response"


class PLD_PredictInput(PredictInput):
    pass

class MaskRcnnInference4Classes(ModelInference):
    model = None
    transforms = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()

    @staticmethod
    def description():
        return model_meta

    def __init__(self, job: Job | None):
        super().__init__(job)

    @classmethod
    def warm_up(
        cls,
        proba_threshold=0.5,
        score_threshold=0.65,
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        num_classes=5,
        model_path="./models/2025-03-03_17-20-57_04m_material_RGB_Instance_Segmentation_best_loss.pt",
    ):
        cls.proba_threshold = proba_threshold
        cls.score_threshold = score_threshold
        cls.transforms = weights.transforms()
        model = maskrcnn_resnet50_fpn(weights=weights)

        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        # Replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=cls.DEVICE, weights_only=True))
        cls.model = model.eval()
        
        with open("ai_inference/inference/maskrcnn_4classes/categories.json") as f:
            data = json.load(f)
        
        cls.categories = [d["name"] for d in data["categories"]]

    def _do_predict(self, input_file: str, input_parameters: PredictInput) -> gpd.GeoDataFrame:
        # font_path = os.path.abspath("./ai_inference/inference/maskrcnn_resnet50/02587_ARIALMT.ttf")
        with Image.open(input_file).convert("RGB") as image:
            # trans_image = self.transforms(image).unsqueeze(0)
            inference_transform = preprocessing.get_transform(train=False)

            # Resize images and targets
            resized_image = preprocessing.resize(image)

            original_width, original_height = image.size
            resized_width, resized_height = resized_image.size

            # Calculate scaling factors
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height

            # Apply additional transformations
            trans_image = inference_transform(resized_image).unsqueeze(0).to(self.DEVICE)

            output = MaskRcnnInference4Classes.model(trans_image)  # type: ignore

            out = output[0]

            boxes = out["boxes"][out["scores"] > self.score_threshold]
            resized_boxes = boxes.clone()
            resized_boxes[:, [0, 2]] /= scale_x  # Scale x coordinates
            resized_boxes[:, [1, 3]] /= scale_y  # Scale y coordinates
            labels = out["labels"][out["scores"] > self.score_threshold]
            labels = [str(label.item()) for label in labels]
            scores = out["scores"][out["scores"] > self.score_threshold]
            
            print(out["scores"].detach().numpy())
            print(scores)

            geometries = [box(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in resized_boxes]
            
            cats = [MaskRcnnInference4Classes.categories[int(l)-1] for l in labels]

            data = {'geometry': geometries, 'label': labels, "categories": cats, 'score': scores.detach().numpy()}

            gdf = gpd.GeoDataFrame(pd.DataFrame(data))

            return gdf

@router.post(f"/{model_meta.name}/", status_code=201)
async def call_predict(input: PLD_PredictInput, request: Request, response: Response) -> dict:
    from ai_inference.main import add_new_job_to_queue

    job: Job = add_new_job_to_queue(request)

    inference = MaskRcnnInference4Classes(job)
    asyncio.create_task(inference.predict(input))

    response.headers["location"] = urljoin(str(request.base_url), f"jobs/{job.job_id}")
    return {"job_id": job.job_id, "status": job.status, "msg": job.msg}


if __name__ == '__main__':
    # python -m ai_inference.inference.maskrcnn_4classes.inference
    from pyproj import CRS as PyprojCRS

    job = None
    input_file = "ai_inference/test_images/Plot_4_.png"
    parameters = ""
    inference = MaskRcnnInference4Classes(job)
    inference.warm_up()
    result_gdf = inference._do_predict(input_file, parameters)
    epsg_code = 4326
    crs = PyprojCRS.from_epsg(epsg_code)
    result_gdf = result_gdf.set_crs(crs)
    result_gdf.to_file("Shape_files/maskrcnn4_bboxes_with_metadata.shp")
