# Importing all necessary libraries
from ast import arg, parse
import cv2
import os
import argparse
from tqdm import tqdm

def convert_all_videos_in_folder(video_folder, output_folder):
    video_files = [os.path.join(video_folder, file) for file in os.listdir(video_folder) if file.endswith(('.MP4', '.avi', '.mkv'))]
    video_to_images(video_files, output_folder)

def video_to_images(video_paths, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for video_path in tqdm(video_paths):
        # Open the video file
        vidcap = cv2.VideoCapture(video_path)
        
        # Frame counter
        count = 0
        
        # Read the video frame by frame
        while True:
            success, image = vidcap.read()
            if not success:
                break
            
            # Save frame as an image
            image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{count:04d}.jpg")
            cv2.imwrite(image_path, image)
            count += 1
        
        # Release the video capture object
        vidcap.release()

video_folder_path = '/home/ivpg/Lacie/Datasets/SfM/video/top'  # Replace with the path to your video folder
output_directory = '/home/ivpg/Lacie/Datasets/SfM/images'  # Replace with the desired output folder path

convert_all_videos_in_folder(video_folder_path, output_directory)


