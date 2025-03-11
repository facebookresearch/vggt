import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import os
import socket
import webbrowser
sys.path.append('vggt/')
import shutil
from datetime import datetime
from demo_hf import demo_fn #, initialize_model
from omegaconf import DictConfig, OmegaConf
import glob
import gc
import time
from viser_fn import viser_wrapper
from gradio_util import demo_predictions_to_glb
from hydra.utils import instantiate
# import spaces
from vggt.models.vggt import VGGT



print("Loading model")

cfg_file = "config/base.yaml"
cfg = OmegaConf.load(cfg_file)
# vggt_model = instantiate(cfg, _recursive_=False)
vggt_model = VGGT()
# Reload vggt_model
# _VGGT_URL = "https://huggingface.co/facebook/vggt_alpha/resolve/main/vggt_alpha_v0.pt"
# pretrain_model = torch.hub.load_state_dict_from_url(_VGGT_URL)


if True:
    _CKPT_PATH = "/fsx-repligen/jianyuan/cvpr2025_ckpts/r518_t7_cmh_v7_0-d4w770q_model_converted.pt"
    pretrain_model = torch.load(_CKPT_PATH)

    # import pdb; pdb.set_trace()
    if "model" in pretrain_model:
        model_dict = pretrain_model["model"]
        vggt_model.load_state_dict(model_dict, strict=False)
    else:
        vggt_model.load_state_dict(pretrain_model, strict=True)


print("Model loaded")

# @torch.inference_mode()

# @spaces.GPU(duration=120)
def vggt_demo(
    input_video,
    input_image,
    conf_thres=3.0,
    frame_filter="all",
    mask_black_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression"
):
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    target_dir = f"input_images_{timestamp}"
    if os.path.exists(target_dir): 
        shutil.rmtree(target_dir)

    os.makedirs(target_dir)
    target_dir_images = target_dir + "/images"
    os.makedirs(target_dir_images)


    if input_video is not None:            
        if not isinstance(input_video, str):
            input_video = input_video["video"]["path"]
    
    cfg_file = "config/base.yaml"
    cfg = OmegaConf.load(cfg_file)

    if input_image is not None:
        input_image = sorted(input_image)        
        for file_name in input_image:
            shutil.copy(file_name, target_dir_images)
    elif input_video is not None:
        vs = cv2.VideoCapture(input_video)

        fps = vs.get(cv2.CAP_PROP_FPS)

        frame_rate = 1
        frame_interval = int(fps * frame_rate)
        
        video_frame_num = 0
        count = 0 
        
        while True:
            (gotit, frame) = vs.read()
            count +=1

            if not gotit:
                break
            
            if count % frame_interval == 0:
                cv2.imwrite(target_dir_images+"/"+f"{video_frame_num:06}.png", frame)
                video_frame_num+=1                
    else:
        return None, "Uploading not finished or Incorrect input format", None, None
        
    all_files = sorted(os.listdir(target_dir_images))

    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]

    # Update frame_filter choices
    frame_filter_choices = ["All"] + all_files
    
    print(f"Files have been copied to {target_dir_images}")
    cfg.SCENE_DIR = target_dir
    
    print("Running demo_fn")
    with torch.no_grad():
        predictions = demo_fn(cfg, vggt_model)
    predictions["pred_extrinsic_list"] = None
    print("Saving predictions")
    
    prediction_save_path = f"{target_dir}/predictions.npz"
    
    np.savez(prediction_save_path, **predictions)


    glbfile = target_dir + f"/glbscene_{conf_thres}_{frame_filter.replace('.', '_')}_mask{mask_black_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb"

    
    glbscene = demo_predictions_to_glb(predictions, conf_thres=conf_thres, filter_by_frames=frame_filter, mask_black_bg=mask_black_bg, show_cam=show_cam, mask_sky=mask_sky, target_dir=target_dir, prediction_mode=prediction_mode)
    glbscene.export(file_obj=glbfile) 

    del predictions
    gc.collect()
    torch.cuda.empty_cache()
    
    print(input_image)
    print(input_video)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
    # Return None for the 3D vggt_model (since we're using viser) and the viser URL
    # viser_url = f"Viser visualization is ready at: http://localhost:{viser_port}"
    # print(viser_url)  # Debug print
    log = "Reconstruction Success. Waiting for visualization."
    return glbfile, log, target_dir, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)

def clear_fields():
    """
    Return None for reconstruction_output and target_dir_output
    (and optionally reset frame_filter to "All" or something else if needed).
    """
    return None, None



def update_log():
    """
    Return None for reconstruction_output and target_dir_output
    (and optionally reset frame_filter to "All" or something else if needed).
    """
    return "Loading and Reconstructing..."



def update_visualization(target_dir, conf_thres, frame_filter, mask_black_bg, show_cam, mask_sky, prediction_mode):
    # Return early if target_dir is None, "None", empty string, or otherwise invalid
    # Check if the predictions file exists
    predictions_path = f"{target_dir}/predictions.npz"

    if target_dir is None or target_dir == "None" or target_dir == "" or not os.path.isdir(target_dir):
        return None, f"No reconstruction available. Please click the Reconstruct button first.", None
    
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first.", None
    
    loaded = np.load(predictions_path, allow_pickle=True)
    # predictions = np.load(f"{target_dir}/predictions.npz", allow_pickle=True)
    # predictions["arr_0"]
    # for key in predictions.files: print(key)
    predictions = {key: loaded[key] for key in loaded.keys()}

    glbfile = target_dir + f"/glbscene_{conf_thres}_{frame_filter.replace('.', '_')}_mask{mask_black_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb"

    if not os.path.exists(glbfile):
        glbscene = demo_predictions_to_glb(predictions, conf_thres=conf_thres, filter_by_frames=frame_filter, mask_black_bg=mask_black_bg, show_cam=show_cam, mask_sky=mask_sky, target_dir=target_dir, prediction_mode=prediction_mode)
        glbscene.export(file_obj=glbfile) 
    return glbfile, "Updating Visualization", target_dir








statue_video = "examples/videos/statue_video.mp4"

apple_video = "examples/videos/apple_video.mp4"
british_museum_video = "examples/videos/british_museum_video.mp4"
cake_video = "examples/videos/cake_video.mp4"
bonsai_video = "examples/videos/bonsai_video.mp4"
face_video =  "examples/videos/in2n_face_video.mp4"
counter_video =  "examples/videos/in2n_counter_video.mp4"

horns_video = "examples/videos/llff_horns_video.mp4"
person_video = "examples/videos/in2n_person_video.mp4"

flower_video = "examples/videos/llff_flower_video.mp4"

fern_video = "examples/videos/llff_fern_video.mp4"

drums_video = "examples/videos/drums_video.mp4"

kitchen_video = "examples/videos/kitchen_video.mp4"

room_video = "examples/videos/room_video.mp4"

# Add the new video examples
single_video = "examples/videos/single.mp4"
single_cartoon_video = "examples/videos/single_cartoon.mp4"
single_oil_painting_video = "examples/videos/single_oil_painting.mp4"

###########################################################################################
apple_images = glob.glob(f'examples/apple/images/*')
bonsai_images = glob.glob(f'examples/bonsai/images/*')
cake_images = glob.glob(f'examples/cake/images/*')
british_museum_images = glob.glob(f'examples/british_museum/images/*')
face_images = glob.glob(f'examples/in2n_face/images/*')
counter_images = glob.glob(f'examples/in2n_counter/images/*')

horns_images = glob.glob(f'examples/llff_horns/images/*')

person_images = glob.glob(f'examples/in2n_person/images/*')
flower_images = glob.glob(f'examples/llff_flower/images/*')

fern_images = glob.glob(f'examples/llff_fern/images/*')
statue_images = glob.glob(f'examples/statue/images/*')

drums_images = glob.glob(f'examples/drums/images/*')
kitchen_images = glob.glob(f'examples/kitchen/images/*')
room_images = glob.glob(f'examples/room/images/*')

single_images = glob.glob(f'examples/single/images/*')
single_cartoon_images = glob.glob(f'examples/single_cartoon/images/*')
single_oil_painting_images = glob.glob(f'examples/single_oil_painting/images/*')
###########################################################################################



# theme = gr.themes.Base()
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(theme=theme,
    css="""
    .custom-log {
        font-style: italic;
        font-size: 1.2em;
        color: #555;
        text-align: center;  /* Centers the text */
    }

    
    /* The container that wraps the radio buttons */
    #my_radio .wrap {
        display: flex;           /* horizontal layout */
        flex-wrap: nowrap;       /* keep them in one row */
        justify-content: center; /* center the group in the row */
        align-items: center;     /* vertically center items if heights differ */
    }

    /* Each radio option: force half-width and center its contents (the circle & label text) */
    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """) as demo:    
    gr.Markdown("""
    # 🏛️ VGGT: Visual Geometry Grounded Transformer
    
    <div style="font-size: 16px; line-height: 1.5;">
    <p><strong>Alpha version</strong> (under fast change)</p>
        
    <p>Upload a video or images to create a 3D reconstruction. Once your media appears in the left panel, click the "Reconstruct" button to begin processing.</p>
    
    <h3>Usage Tips:</h3>
    <ol>
        <li>After reconstruction, you can adjust the visualization by adjusting the confidence threshold, selecting specific frames to display, and so on.</li>
        <li>Performance note: While the model itself processes quickly (~0.2 seconds), initial setup and visualization may take longer. First-time use requires downloading model weights, and rendering dense point clouds can be resource-intensive.</li>
        <li>Known limitation: The model currently exhibits weird behavior with videos centered around human subjects. This issue is being addressed in upcoming updates.</li>
    </ol>
    </div>
    """)

    
    # Add a hidden textbox for target_dir with default value "None"
    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")


    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)
            
            
        with gr.Column(scale=3):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses; Zoom in to see details)**")
                log_output = gr.Markdown("Please upload a video or images, and then click the Reconstruct button to start.", elem_classes=["custom-log"])
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)


            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                # revisual_btn = gr.Button("Update Visualization", scale=1)
                clear_btn = gr.ClearButton([input_video, input_images, reconstruction_output, log_output, target_dir_output], scale=1) #Modified reconstruction_output
            
    

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch", ], 
                    label="Select a Prediction Mode (using which branch to predict point cloud):", 
                    value="Depthmap and Camera Branch",
                    scale=1,
                    elem_id="my_radio",  # <-- references the CSS above
                )
            
            
            # Move these controls to a new row above the log output
            with gr.Row():
                conf_thres = gr.Slider(minimum=0.1, maximum=10.0, value=2.0, step=0.1, label="Conf Thres")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
    
    
    examples = [
        # [single_video, single_images, 0.5, "All", False, True, False],
        [single_cartoon_video, single_cartoon_images, 0.5, "All", False, True, False, "Depthmap and Camera Branch"],
        [single_oil_painting_video, single_oil_painting_images, 0.5, "All", False, True, True, "Depthmap and Camera Branch"],
        [room_video, room_images, 1.1, "All", False, True, False, "Depthmap and Camera Branch"],
        [counter_video, counter_images, 1.5, "All", False, True, False, "Depthmap and Camera Branch"],
        [flower_video, flower_images, 1.5, "All", False, True, False, "Depthmap and Camera Branch"],
        [kitchen_video, kitchen_images, 3, "All", False, True, False, "Depthmap and Camera Branch"],
        [fern_video, fern_images, 1.5, "All", False, True, False, "Depthmap and Camera Branch"],
        # Add the new examples
    ]
    
    gr.Examples(examples=examples, 
                inputs=[input_video, input_images, conf_thres, frame_filter, mask_black_bg, show_cam, mask_sky, prediction_mode],
                outputs=[reconstruction_output, log_output, target_dir_output, frame_filter],
                fn=vggt_demo,
                cache_examples=False,
                examples_per_page=50,
                )


    submit_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[reconstruction_output, target_dir_output]
    ).then(
        fn=update_log,
        inputs=[],
        outputs=[log_output]
    ).then(
        fn=vggt_demo,
        inputs=[input_video, input_images, conf_thres, frame_filter, mask_black_bg, show_cam, mask_sky, prediction_mode],
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter]
    )
    
    
    # Add event handlers for automatic updates when parameters change
    conf_thres.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output, target_dir_output],
    )
    
    frame_filter.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output, target_dir_output],
    )
    
    mask_black_bg.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output, target_dir_output],
    )
    
    show_cam.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output, target_dir_output],
    )
    
    mask_sky.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output, target_dir_output],
    )
    
    prediction_mode.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output, target_dir_output],
    )
    
    demo.queue(max_size=20).launch(show_error=True, share=True) #, share=True, server_port=7888, server_name="0.0.0.0")
    
    
    # share=True
    # demo.queue(max_size=20, concurrency_count=1).launch(debug=True, share=True)
########################################################################################################################

