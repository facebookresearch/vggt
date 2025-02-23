try:
    import os
    
    import trimesh
    import open3d as o3d

    import gradio as gr
    import numpy as np
    import matplotlib
    from scipy.spatial.transform import Rotation

    print("Successfully imported the packages for Gradio visualization")
except:
    print(
        f"Failed to import packages for Gradio visualization. Please disable gradio visualization"
    )


def visualize_by_gradio(glbfile):
    """
    Set up and launch a Gradio interface to visualize a GLB file.

    Args:
        glbfile (str): Path to the GLB file to be visualized.
    """

    def load_glb_file(glb_path):
        # Check if the file exists and return the path or error message
        if os.path.exists(glb_path):
            return glb_path, "3D Model Loaded Successfully"
        else:
            return None, "File not found"

    # Load the GLB file initially to check if it's valid
    initial_model, log_message = load_glb_file(glbfile)

    # Create the Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# GLB File Viewer")

        # 3D Model viewer component
        model_viewer = gr.Model3D(
            label="3D Model Viewer", height=600, value=initial_model
        )

        # Textbox for log output
        log_output = gr.Textbox(label="Log", lines=2, value=log_message)

    # Launch the Gradio interface
    demo.launch(share=True)


def vggsfm_predictions_to_glb(predictions) -> trimesh.Scene:
    """
    Converts VGG SFM predictions to a 3D scene represented as a GLB.

    Args:
        predictions (dict): A dictionary containing model predictions.

    Returns:
        trimesh.Scene: A 3D scene object.
    """
    # Convert predictions to numpy arrays
    vertices_3d = predictions["points3D"].cpu().numpy()
    colors_rgb = (predictions["points3D_rgb"].cpu().numpy() * 255).astype(
        np.uint8
    )


    if True:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        filtered_pcd = pcd.select_by_index(ind)

        print(f"Filter out {len(vertices_3d) - len(filtered_pcd.points)} 3D points")
        vertices_3d = np.asarray(filtered_pcd.points)
        colors_rgb = np.asarray(filtered_pcd.colors).astype(np.uint8)

    

    camera_matrices = predictions["extrinsics_opencv"].cpu().numpy()

    # Calculate the 5th and 95th percentiles along each axis
    lower_percentile = np.percentile(vertices_3d, 5, axis=0)
    upper_percentile = np.percentile(vertices_3d, 95, axis=0)

    # Calculate the diagonal length of the percentile bounding box
    scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(
        vertices=vertices_3d, colors=colors_rgb
    )
    
    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    # Add camera models to the scene
    for i in range(num_cameras):
        world_to_camera = extrinsics_matrices[i]
        camera_to_world = np.linalg.inv(world_to_camera)
        rgba_color = colormap(i / num_cameras)
        current_color = tuple(int(255 * x) for x in rgba_color[:3])

        integrate_camera_into_scene(
            scene_3d, camera_to_world, current_color, scene_scale
        )

    # Align scene to the observation of the first camera
    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

    return scene_3d


def apply_scene_alignment(
    scene_3d: trimesh.Scene, extrinsics_matrices: np.ndarray
) -> trimesh.Scene:
    """
    Aligns the 3D scene based on the extrinsics of the first camera.

    Args:
        scene_3d (trimesh.Scene): The 3D scene to be aligned.
        extrinsics_matrices (np.ndarray): Camera extrinsic matrices.

    Returns:
        trimesh.Scene: Aligned 3D scene.
    """
    # Set transformations for scene alignment
    opengl_conversion_matrix = get_opengl_conversion_matrix()

    # Rotation matrix for alignment (180 degrees around the y-axis)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler(
        "y", 180, degrees=True
    ).as_matrix()

    # Apply transformation
    initial_transformation = (
        np.linalg.inv(extrinsics_matrices[0])
        @ opengl_conversion_matrix
        @ align_rotation
    )
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def integrate_camera_into_scene(
    scene: trimesh.Scene,
    transform: np.ndarray,
    face_colors: tuple,
    scene_scale: float,
):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler(
        "z", 45, degrees=True
    ).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler(
        "z", 2, degrees=True
    ).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(
        complete_transform, vertices_combined
    )

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(
        vertices=vertices_transformed, faces=mesh_faces
    )
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def transform_points(
    transformation: np.ndarray, points: np.ndarray, dim: int = None
) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(
        -1, -2
    )  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix