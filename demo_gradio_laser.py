# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import json  # <-- [ADDED] for writing cameras.json


sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))


model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# small helpers to summarize cameras
# -------------------------------------------------------------------------
def _mat_to_euler_xyz_deg(R):
    """
    Convert a rotation matrix to XYZ (roll, pitch, yaw) in degrees.
    Right-handed, OpenCV-like. Minimal, gimbal-lock tolerant.
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def _cameras_list(preds):
    """
    Build a per-frame camera dictionary list from predictions.
    Accepts extrinsic as (S,3,4) or (S,4,4). Intrinsic is (S,3,3).
    """
    E = preds["extrinsic"]     # (S,3,4) or (S,4,4)
    K = preds["intrinsic"]     # (S,3,3)
    names = preds.get("image_names", [f"{i:06d}.png" for i in range(len(E))])

    cams = []
    for i in range(len(E)):
        Ei = E[i]
        # --- make 4x4 homogenous ---
        if Ei.shape == (3, 4):
            H = np.eye(4, dtype=Ei.dtype)
            H[:3, :4] = Ei
        elif Ei.shape == (4, 4):
            H = Ei
        else:
            raise ValueError(f"Unexpected extrinsic shape {Ei.shape}; expected (3,4) or (4,4)")

        # VGGT extrinsic is camera-from-world (C = E * Xw). Invert to get world-from-camera.
        Twc = np.linalg.inv(H)
        Rwc, t = Twc[:3, :3], Twc[:3, 3]
        roll, pitch, yaw = _mat_to_euler_xyz_deg(Rwc)

        # intrinsic sanity (still logable if present as expected)
        fx = float(K[i][0, 0]); fy = float(K[i][1, 1])
        cx = float(K[i][0, 2]); cy = float(K[i][1, 2])

        cams.append({
            "index": int(i),
            "image": os.path.basename(names[i]) if i < len(names) else f"{i:06d}.png",
            "position_m": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
            "euler_xyz_deg": {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        })
    return cams


def _cameras_text(cams):
    ##these are one-liners to be read in the terminal
    lines = []
    for c in cams:
        p = c["position_m"]
        r = c["euler_xyz_deg"]
        k = c["intrinsics"]
        lines.append(
            f'Camera {c["index"]} ({c["image"]}): '
            f'pos=[{p["x"]:.3f},{p["y"]:.3f},{p["z"]:.3f}], '
            f'rpy=[{r["roll"]:.1f},{r["pitch"]:.1f},{r["yaw"]:.1f}], '
            f'fx={k["fx"]:.1f}, fy={k["fy"]:.1f}, cx={k["cx"]:.1f}, cy={k["cy"]:.1f}'
        )
    return "\n".join(lines)


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    print(f"Processing images from {target_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")
    model = model.to(device); model.eval()

    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    t0_load = time.time()
    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape} (load+preproc {(time.time()-t0_load):.3f}s)")

    # ---------- inference timing ----------
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print("Running inference...")
    t0_inf = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    t_inf = time.time() - t0_inf
    print(f"Inference time: {t_inf:.3f}s")

    # ---------- pose decode timing ----------
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    t0_pose = time.time()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    t_pose = time.time() - t0_pose
    print(f"Pose decode time: {t_pose:.3f}s")

    # Convert tensors to numpy (so camera math is on CPU/NumPy)
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions["pose_enc_list"] = None
    predictions["image_names"] = image_names  # keep names for per-frame association

    # ---------- camera extraction timing ----------
    t0_cam = time.time()
    cams = _cameras_list(predictions)   # uses padded 4x4 + inverse
    t_cam = time.time() - t0_cam
    print(f"Camera extraction time: {t_cam:.3f}s for {len(cams)} frames "
          f"(~{(1000*t_cam/max(1,len(cams))):.1f} ms/cam)")

    # Save per-frame cameras to JSON
    try:
        cams_path = os.path.join(target_dir, "cameras.json")
        with open(cams_path, "w") as f:
            json.dump(cams, f, indent=2)
        print(f"Wrote per-frame camera info to {cams_path}")
    except Exception as e:
        print(f"Warning: could not write cameras.json ({e})")

    # (Optional; not part of timing you asked for)
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    # [ADDED] Build a readable camera summary for the log pane
    try:
        cams = _cameras_list(predictions)
        cams_text = _cameras_text(cams)
        cams_hint = "\n\nPer-frame cameras written to cameras.json."
    except Exception as e:
        cams_text = f"(Could not summarize cameras: {e})"
        cams_hint = ""

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = (
        f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."
        f"{cams_hint}\n\n{cams_text}"
    )

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# Example images
# -------------------------------------------------------------------------

great_wall_video = "examples/videos/great_wall.mp4"
colosseum_video = "examples/videos/Colosseum.mp4"
room_video = "examples/videos/room.mp4"
kitchen_video = "examples/videos/kitchen.mp4"
fern_video = "examples/videos/fern.mp4"
single_cartoon_video = "examples/videos/single_cartoon.mp4"
single_oil_painting_video = "examples/videos/single_oil_painting.mp4"
pyramid_video = "examples/videos/pyramid.mp4"


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    # Instead of gr.State, we use a hidden Textbox:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    gr.HTML(
        """
    <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. VGGT takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left to provide your input. Videos will be automatically split into individual frames (one frame per second).</li>
        <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
        <li>
        <strong>Adjust Visualization (Optional):</strong>
        After reconstruction, you can fine-tune the visualization using the options below
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
            <ul>
            <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
            <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
            <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
            <li><em>Filter Sky / Filter Black Background:</em> Remove sky or black-background points.</li>
            <li><em>Select a Prediction Mode:</em> Choose between "Depthmap and Camera Branch" or "Pointmap Branch."</li>
            </ul>
        </details>
        </li>
    </ol>
    <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, which are independent of VGGT's processing time. </span></p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # ---------------------- Examples section ----------------------
    examples = [
        [colosseum_video, "22", None, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [pyramid_video, "30", None, 35.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_cartoon_video, "1", None, 15.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_oil_painting_video, "1", None, 20.0, False, False, True, True, "Depthmap and Camera Branch", "True"],
        [room_video, "8", None, 5.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [kitchen_video, "25", None, 50.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [fern_video, "20", None, 45.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
    ]

    def example_pipeline(
        input_video,
        num_images_str,
        input_images,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        is_example_str,
    ):
        """
        1) Copy example images to new target_dir
        2) Reconstruct
        3) Return model3D + logs + new_dir + updated dropdown + gallery
        We do NOT return is_example. It's just an input.
        """
        target_dir, image_paths = handle_uploads(input_video, input_images)
        # Always use "All" for frame_filter in examples
        frame_filter = "All"
        glbfile, log_msg, dropdown = gradio_demo(
            target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
        )
        return glbfile, log_msg, target_dir, dropdown, image_paths

    gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

    gr.Examples(
        examples=examples,
        inputs=[
            input_video,
            num_images,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery],
        fn=example_pipeline,
        cache_examples=False,
        examples_per_page=50,
    )

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    #  - Then set is_example = "False"
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        outputs=[reconstruction_output, log_output, frame_filter],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
    conf_thres.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    frame_filter.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_black_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_white_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    show_cam.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_sky.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    prediction_mode.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)
