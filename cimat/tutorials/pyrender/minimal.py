import pyrender
import numpy as np
import matplotlib.pyplot as plt

from utils.viz_utils import scene_from_vggt, add_vggt_glb
from utils.git_utils import GIT_ROOT


def vggt_extrinsics_to_gitf(extrinsics: np.ndarray) -> np.ndarray:
    # world_to_cv_cam = 3x4 VGGT extrinsic
    world_to_cv_cam = extrinsics
    cv_cam_to_world = np.eye(4)
    cv_cam_to_world[:3, :] = world_to_cv_cam
    cv_cam_to_world = np.linalg.inv(cv_cam_to_world)

    # convert to OpenGL-style camera-to-world
    cv_to_gl = np.diag([1, -1, -1, 1])  # CV cam → OpenGL cam
    gl_cam_to_world = cv_cam_to_world @ cv_to_gl
    return gl_cam_to_world


def main():
    glb_file = f"{GIT_ROOT}/data/vggt_output/input_images_20250622_102110_094677/glbscene_50_All_maskbFalse_maskwFalse_camTrue_skyFalse_predDepthmap_and_Camera_Branch.glb"
    predictions_file = f"{GIT_ROOT}/data/vggt_output/input_images_20250622_102110_094677/predictions.npz"
    predictions = np.load(predictions_file, 'r')
    #['pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf', 'images', 'extrinsic', 'intrinsic', 'world_points_from_depth']

    # Step 4: Create PyRender scene
    scene = scene_from_vggt(glb_file)

    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
    gl_cam_to_world = vggt_extrinsics_to_gitf(predictions["extrinsic"][1])
    scene.add(camera, pose=gl_cam_to_world)
    
    light = pyrender.SpotLight(
        color = np.ones(3),
        intensity = 10.0,
    )
    pl = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    #scene.add(light, pose = gl_cam_to_world)
    scene.add(pl)
    r = pyrender.OffscreenRenderer(400, 400)
    color, depth = r.render(scene)
    plt.imshow(color)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
