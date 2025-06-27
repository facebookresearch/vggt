import os

from PIL import Image

from sim.utils.git_utils import GIT_ROOT


DATA_PATH = f"{GIT_ROOT}/.runs/data/"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


class PredictionData:
    id = 0
    def __init__(self, image_path_list: list) -> None:
        if len(image_path_list) == 0:
            raise ValueError("At least 1 image is required")
        
        self.images = [Image.open(image_path) for image_path in image_path_list]

        self.image_names = [
            os.path.basename(image_path)
            for image_path in image_path_list
        ]

        self.target_dir = f"{DATA_PATH}/input_images_{PredictionData.id}"
        PredictionData.id += 1
        os.makedirs(f"{self.target_dir}/images/")
        for image, name in zip(self.images, self.image_names):
            path = f"{self.target_dir}/images/{name}"
            image.save(path)

        self.predictions = None
        self.glb_points = None

    def __getattr__(self, key: str) -> object:
        if not key in self.predictions:
            raise ValueError(f"{key} not found")
        return self.predictions[key]

