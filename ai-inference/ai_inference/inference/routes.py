from typing import List

from pydantic import BaseModel
from fastapi import APIRouter

from ai_inference.inference.common import ModelDescription, ModelInference

router = APIRouter(prefix="/models", tags=["models"])
model_inferences: List[ModelInference] = []


class ModelInferencesResponse(BaseModel):
    models: List[ModelDescription]


def add_model_inference_class(model_class: ModelInference):
    model_inferences.append(model_class)
    model_class.warm_up()


@router.get("/")
def models() -> ModelInferencesResponse:
    return {"models": [model.description() for model in model_inferences]}
