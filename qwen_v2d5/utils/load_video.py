#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: load_video.py
@brief: Utility to load a single frame from a video file as a PIL Image.
@author: Matha Goram
@date: 2025-07-21
@license: MIT License
@version: 1.0
@description: This script provides a function to load a specific frame from a video file
as a PIL Image. It handles errors such as file not found, video not opening, and frame out of bounds.

"""
import cv2
from PIL import Image
import os

def load_frame_as_pil(video_path, frame_number=0):
    """
    Loads a single frame from a video file as a PIL Image.

    Args:
        video_path (str): The path to the video file.
        frame_number (int): The index of the frame to extract (default is 0 for the first frame).

    Returns:
        PIL.Image.Image or None: The extracted frame as a PIL Image, or None if an error occurs.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_number >= total_frames or frame_number < 0:
        print(f"Error: Frame number {frame_number} is out of bounds. Video has {total_frames} frames.")
        cap.release()
        return None

    # Set the video to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if ret:
        # OpenCV reads images in BGR format, so convert to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        print(f"Successfully loaded frame {frame_number} from {video_path}")
        return pil_image
    else:
        print(f"Error: Could not read frame {frame_number} from {video_path}")
        return None

    cap.release() # Release the video capture object

if __name__ == "__main__":
    # Example Usage:
    # Create a dummy video file for testing if you don't have one
    # You can use a tool like ffmpeg to create a short dummy video:
    # ffmpeg -f lavfi -i "color=c=blue:s=1280x720:d=5" -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Test Video':x=(w-text_w)/2:y=(h-text_h)/2:fontsize=50:fontcolor=white" test_video.mp4

    dummy_video_path = "../../videos/egll-09r.mp4"  # Replace with your video file path

    # Try to create a dummy video if it doesn't exist
    if not os.path.exists(dummy_video_path):
        print(f"Creating a dummy video file: {dummy_video_path}")
        try:
            # This is a placeholder for actual video creation.
            # You'd typically use a tool like ffmpeg from your terminal.
            # For a programmatic approach, you might need more complex libraries
            # or pre-generate a video manually for testing.
            # For this example, we'll just indicate it's missing.
            print("Please create 'test_video.mp4' manually for testing, e.g., using ffmpeg.")
            print(f"Example: ffmpeg -f lavfi -i \"color=c=blue:s=640x480:d=1\" -pix_fmt yuv420p {dummy_video_path}")

        except Exception as e:
            print(f"Could not create dummy video: {e}")

    if os.path.exists(dummy_video_path):
        # Load the first frame
        first_frame = load_frame_as_pil(dummy_video_path, frame_number=0)
        if first_frame:
            print(f"First frame details: {first_frame.size}, {first_frame.mode}")
            # You can then save or display the image
            # first_frame.save("first_frame.jpg")
            # first_frame.show()

        print("-" * 30)

        # Load a different frame (e.g., frame 30)
        specific_frame = load_frame_as_pil(dummy_video_path, frame_number=30)
        if specific_frame:
            print(f"Frame 30 details: {specific_frame.size}, {specific_frame.mode}")
            # specific_frame.save("frame_30.jpg")
            # specific_frame.show()
    else:
        print(f"Skipping example usage as {dummy_video_path} does not exist.")
