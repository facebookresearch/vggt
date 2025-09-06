import pyrender

from utils.viz_utils import scene_from_vggt, add_vggt_glb
from utils.git_utils import GIT_ROOT


def main():
    glb_file_1 = f"{GIT_ROOT}/data/vggt_output/input_images_1/glbscene.glb"
    #glb_file_2 = f"{GIT_ROOT}/data/vggt_output/input_images_20250622_102223_545756/glbscene_50_All_maskbFalse_maskwFalse_camTrue_skyFalse_predDepthmap_and_Camera_Branch.glb" #Need to reset worl frame.
    scene = pyrender.Scene()
    add_vggt_glb(scene, glb_file_1)
    #add_vggt_glb(scene, glb_file_2, scale=1.21)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0))
    pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == '__main__':
    main()
