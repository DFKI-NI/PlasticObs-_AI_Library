
import os

def create_file_with_content():
    # Ask the user for the folder name
    subfolder_structure = "./ai_inference/inference/"
    folder_name = input("Enter the folder name (use underscores '_' instead of hyphens '-' or spaces): ")
     
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Define the file name
    file_name = "inference.py"
    file_path = os.path.join(subfolder_structure, folder_name, file_name)

    # Prewritten content to be added to the file
    prewritten_content = """\
    # add your libraries

    # handle geopandas stuff for shapefile upload
    import geopandas as gpd
    from shapely.geometry import Point

    # handle API stuff
    from ai_inference.inference.common import ModelDescription, ModelInference, PredictInput
    from ai_inference.inference.maskrcnn_resnet50 import preprocessing
    from ai_inference.inference.routes import router
    from ai_inference.job import Job

    model_meta = ModelDescription(
        name="maskrcnn_resnet50",
        description=\"\"\"\\
        Description of my model
        \"\"\",
        )
    
    class MyInference(ModelInference): # make sure to rename our Inference class
        model = None

        @staticmethod
        def description():
            return model_meta

        def __init__(self, job: Job | None):
            super().__init__(job)

        @classmethod
        def warm_up(
            cls,
            model_name="my_modeL_file.pt",
        ): # extend with other stuff
        # initilize your model
        pass
        
        def _do_predict(self, input_file: str, input_parameters: PredictInput) -> gpd.GeoDataFrame:
            with Image.open(input_file).convert("RGB") as image:
                # do your image processing
                pass
            
            # create data dictionary with geometries and other important stuff
            
            # return data to the GeoNode
            gdf = gpd.GeoDataFrame(pd.DataFrame(data))

            return gdf

    @router.post(f"/{model_meta.name}/", status_code=201)
    async def call_predict(input: PLD_PredictInput, request: Request, response: Response) -> dict:
        from ai_inference.main import add_new_job_to_queue

        job: Job = add_new_job_to_queue(request)

        inference = MaskRcnnInference(job)
        asyncio.create_task(inference.predict(input))

        response.headers["location"] = urljoin(str(request.base_url), f"jobs/{job.job_id}")
        return {"job_id": job.job_id, "status": job.status, "msg": job.msg}


    if __name__ == '__main__':
        pass
    """

    # Create the file in the specified folder and write the prewritten content
    with open(file_path, 'w') as file:
        file.write(prewritten_content)

    print(f"File '{file_name}' has been created in the folder '{folder_name}' with prewritten content.")

# Run the function
create_file_with_content()
