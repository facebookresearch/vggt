import cv2
import os

# Parameters
# name = "llff_horns"
# name = "in2n_person"

# name = "llff_flower"
# name = "in2n_person"
# name = "statue"

name = "kitchen"

# name = "drums"

folder_path = f'vggsfm_code/examples/{name}/images'  # Update with the path to your images
video_path = f'vggsfm_code/examples/videos/{name}_video.mp4'
fps = 1  # frames per second

# Get all image files from the directory
images = [img for img in os.listdir(folder_path)]
images.sort()  # Sort the images by name

# Read the first image to get the size
frame = cv2.imread(os.path.join(folder_path, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'x264' for codec
video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# Add images to video
for image in images:
    img = cv2.imread(os.path.join(folder_path, image))
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))  # Resize image to match the first image's size
    video.write(img)

# Release the video writer
video.release()