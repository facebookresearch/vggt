import os
import cv2
import torch
import numpy as np
import gradio as gr
# import spaces
import sys
import os
sys.path.append('vggt/')
import shutil
from datetime import datetime
from demo_hf import demo_fn
from omegaconf import DictConfig, OmegaConf
import glob
import gc
import time

def vggsfm_demo(
    input_video,
    input_image,
):
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    import pdb; pdb.set_trace()
    debug = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    max_input_image = 25

    target_dir = f"input_images_{timestamp}"
    if os.path.exists(target_dir): 
        shutil.rmtree(target_dir)

    os.makedirs(target_dir)
    target_dir_images = target_dir + "/images"
    os.makedirs(target_dir_images)


    if debug:
        predictions = torch.load("predictions_scene2.pth")
    else:
        
        if input_video is not None:            
            if not isinstance(input_video, str):
                input_video = input_video["video"]["path"]
        
        cfg_file = "vggsfm_code/cfgs/demo.yaml"
        cfg = OmegaConf.load(cfg_file)

        if input_image is not None:

            input_image = sorted(input_image)
            input_image = input_image[:max_input_image]
            
            recon_num = len(input_image)
            if recon_num<3:
                return None, "Please input at least three frames"

            # Copy files to the new directory
            for file_name in input_image:
                shutil.copy(file_name, target_dir_images)
        elif input_video is not None:
            vs = cv2.VideoCapture(input_video)

            fps = vs.get(cv2.CAP_PROP_FPS)

            frame_rate = 1
            frame_interval = int(fps * frame_rate)
            
            video_frame_num = 0
            count = 0 
            
            while video_frame_num<max_input_image:
                (gotit, frame) = vs.read()
                count +=1

                if not gotit:
                    break
                
                if count % frame_interval == 0:
                    cv2.imwrite(target_dir_images+"/"+f"{video_frame_num:06}.png", frame)
                    video_frame_num+=1
                    
            recon_num = video_frame_num     
            if recon_num<3:
                return None, "Please input at least three frames"
        else:
            return None, "Uploading not finished or Incorrect input format"
            
        cfg.query_frame_num = query_frame_num
        cfg.max_query_pts = max_query_pts
        print(f"Files have been copied to {target_dir_images}")
        cfg.SCENE_DIR = target_dir
        
        # try:
        predictions = demo_fn(cfg)
        # except:
        # return None, "Something seems to be incorrect. Please verify that your inputs are formatted correctly. If the issue persists, kindly create a GitHub issue for further assistance."
    
    glbscene = vggsfm_predictions_to_glb(predictions)
    
    glbfile = target_dir + "/glbscene.glb"
    glbscene.export(file_obj=glbfile) 
    # glbscene.export(file_obj=glbfile, line_settings= {'point_size': 20})    


    del predictions
    gc.collect()
    torch.cuda.empty_cache()
    
    print(input_image)
    print(input_video)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
    
    # glbscene.geometry['geometry_0'].colors.max()
    # recon_num
    return glbfile, f"Reconstruction complete ({recon_num} frames)"




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



###########################################################################################


with gr.Blocks() as demo:
    
    gr.Markdown("""
    # 🏛️ VGGT: Visual Geometry Grounded Transformer
    
    <div style="font-size: 16px; line-height: 1.2;">
    Alpha version (testing).
    </div>
    """)


    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)
            # num_query_images = gr.Slider(minimum=1, maximum=10, step=1, value=4, label="Number of query images (key frames)",
            #                              info="More query images usually lead to better reconstruction at a lower speed. If the viewpoint differences between your images are minimal, you can set this value to 1. ")
            # num_query_points = gr.Slider(minimum=600, maximum=6000, step=1, value=2048, label="Number of query points",
            #                              info="More query points usually lead to denser reconstruction at a lower speed.")
        
        with gr.Column(scale=3):
            reconstruction_output = gr.Model3D(label="3D Reconstruction (Point Cloud and Camera Poses; Zoom in to see details)", height=520, zoom_speed=0.5, pan_speed=0.5)
            log_output = gr.Textbox(label="Log")

    with gr.Row():
        submit_btn = gr.Button("Reconstruct", scale=1)

        # submit_btn = gr.Button("Reconstruct", scale=1, elem_attributes={"style": "background-color: blue; color: white;"})
        clear_btn = gr.ClearButton([input_video, input_images, num_query_images, num_query_points, reconstruction_output, log_output], scale=1)
    
    
    
    
    examples = [
        [flower_video, flower_images, 2, 4096],
        [kitchen_video, kitchen_images, 4, 2048],
        [person_video, person_images, 3, 2048],
        [statue_video, statue_images, 4, 2048],
        [drums_video, drums_images, 4, 2048],
        [counter_video, counter_images, 4, 2048],
        [fern_video, fern_images, 2, 4096],
        [horns_video, horns_images, 3, 4096],
        [apple_video, apple_images, 6, 2048],
        # [british_museum_video, british_museum_images, 1, 4096],
        [bonsai_video, bonsai_images, 3, 2048],
        # [face_video, face_images, 4, 2048],
        # [cake_video, cake_images, 3, 2048],
    ]
    
    
    
    gr.Examples(examples=examples, 
                inputs=[input_video, input_images, num_query_images, num_query_points],
                outputs=[reconstruction_output, log_output],  # Provide outputs
                fn=vggsfm_demo,  # Provide the function
                cache_examples=False,
                examples_per_page=50,
                )


    submit_btn.click(
        vggsfm_demo,
        [input_video, input_images, num_query_images, num_query_points],
        [reconstruction_output, log_output],
        concurrency_limit=1
    )

    demo.launch(debug=True, share=True)
    # demo.queue(max_size=20).launch(show_error=True, share=True)
    # demo.queue(max_size=20, concurrency_count=1).launch(debug=True, share=True)
########################################################################################################################
