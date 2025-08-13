import numpy as np

from utils.git_utils import GIT_ROOT


def main() -> None:
    #['pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf', 'images', 'extrinsic', 'intrinsic', 'world_points_from_depth']
    predictions_file_1 = f"{GIT_ROOT}/data/vggt_output/input_images_20250622_102110_094677/predictions.npz"
    predictions_1 = np.load(predictions_file_1, 'r')
    predictions_file_2 = f"{GIT_ROOT}/data/vggt_output/input_images_20250622_102223_545756/predictions.npz"
    predictions_2 = np.load(predictions_file_2, 'r')
    print(predictions_1["intrinsic"][0])
    print(predictions_2["intrinsic"][0])
    print(predictions_1["intrinsic"][1])
    print(predictions_2["intrinsic"][1])


if __name__ == '__main__':
    main()
