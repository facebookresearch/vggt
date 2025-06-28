import os
import shutil

from PIL import Image

from sim.utils.git_utils import GIT_ROOT


DATA_PATH = f"{GIT_ROOT}/.runs/data/"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


class PredictionData:
    id = 0
    def __init__(self, *, image_path_list: list = [], target_dir: str="") -> None:
        self.check_args(image_path_list, target_dir)
        if image_path_list != []:    
            images = [Image.open(image_path) for image_path in image_path_list]

            self.image_names = [
                os.path.basename(image_path)
                for image_path in image_path_list
            ]

            self.target_dir = f"{DATA_PATH}/input_images_{PredictionData.id}"
            if os.path.exists(self.target_dir):
                shutil.rmtree(self.target_dir)
            PredictionData.id += 1
            os.makedirs(f"{self.target_dir}/images/")
            for image, name in zip(images, self.image_names):
                path = f"{self.target_dir}/images/{name}"
                image.save(path)
        elif target_dir != "":
            self.target_dir = target_dir
            image_names = glob.glob(os.path.join(target_dir, "images", "*"))
            self.image_names = sorted(image_names)

        self._predictions = None

    def __getattr__(self, key: str) -> object:
        if not key in self.predictions:
            raise ValueError(f"{key} not found")
        return self.predictions[key]

    @property
    def predictions(self) -> dict:
        if self._predictions is not None:
            return self._predictions
    
        predictions_path = f"{self.target_dir}/predictions.npz"
        if not os.path.exists(predictions_path):
            raise ValueError("Need to generate predictions first")
        loaded = np.load(predictions_path)
        self._predictions = {key: np.array(loaded[key]) for key in key_list}

        return self._predictions

    def check_args(self, *args) -> None:
        enabled = sum(not arg for arg in args)
        if enabled != 1:
            print("Only one kwarg must be passed")

    def delete(self) -> None:
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
