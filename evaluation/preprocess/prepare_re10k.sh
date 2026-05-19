# The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

# First download the data from here: https://google.github.io/realestate10k/download.html

# download re10k test sequences by yourself, you can refer to the below script
python evaluation/preprocess/download_re10k.py

# convert camera annotations in metadata to video data folder
python evaluation/preprocess/prepare_re10k.py