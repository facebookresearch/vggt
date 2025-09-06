import trimesh
import pyrender
import numpy as np

# Load the GLB file
GLB_FILE_EXAMPLE = "data/vggt_output/input_images_20250622_102110_094677/glbscene_50_All_maskbFalse_maskwFalse_camTrue_skyFalse_predDepthmap_and_Camera_Branch.glb"


def scene_from_vggt(glb_file: str) -> pyrender.Scene:
    # Create Pyrender scene
    scene = pyrender.Scene()

    add_vggt_glb(scene, glb_file)
    # Add lighting
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0))
    return scene


def add_vggt_glb(scene: pyrender.Scene, glb_file: str, scale:float=1.0) -> pyrender.Scene:
    # Load the GLB file
    data = trimesh.load(glb_file)
    for name, geometry in data.geometry.items():
        if isinstance(geometry, trimesh.Trimesh):
            geometry.apply_scale(scale)
            mesh = pyrender.Mesh.from_trimesh(geometry)
            scene.add(mesh)
        elif isinstance(geometry, trimesh.points.PointCloud):
            points = geometry.vertices*scale
            colors = geometry.colors if hasattr(geometry, 'colors') else np.ones((len(points), 3)) * 128
            mesh = pyrender.Mesh.from_points(points, colors=colors.astype(np.uint8))
            scene.add(mesh)

    return scene

