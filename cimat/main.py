from tqdm import tqdm

from sim.pointclouds.prediction_output import PredictionData
from sim.pointclouds.inference import CudaInference

DATA_PATH = "/home/emmanuel.larralde/Desktop/CIMAT/Semestres/Verano/data/images/guanajuato/basilica"

def main():
    prediction_1 = PredictionData(
        image_path_list = [
            f"{DATA_PATH}/frame_000087.jpg",
            f"{DATA_PATH}/frame_000071.jpg",
        ]
    )

    prediction_2 = PredictionData(
        image_path_list = [
            f"{DATA_PATH}/frame_000071.jpg",
            f"{DATA_PATH}/frame_000506.jpg",
        ]
    )

    prediction_3 = PredictionData(
        image_path_list = [
            f"{DATA_PATH}/frame_000506.jpg",
            f"{DATA_PATH}/frame_000512.jpg"
        ]
    )

    prediction_list = [prediction_1, prediction_2, prediction_3]
    print("Loading model")
    vggt_inference = CudaInference()
    print("Finished loading model")
    print("VGGT Forwarding per scene")
    for prediction in tqdm(prediction_list):
        vggt_inference.run(prediction)
    vggt_inference.clear_model()


if __name__ == '__main__':
    main()
