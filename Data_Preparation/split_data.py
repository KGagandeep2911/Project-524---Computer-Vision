import cv2
import os
import pandas as pd
import json
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def extract_frames(video_path, start_frame, end_frame, output_dir, prefix, img_height, img_width):
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"Error: Couldn't open video {video_path}. Skipping.")
        return

    # Set the starting frame (this is where we will begin reading from)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame from {video_path} at frame {frame_count}. Skipping this frame.")
            break  # End the loop if no frame is read

        # Resize the frame to the desired dimensions
        frame_resized = cv2.resize(frame, (img_width, img_height))

        # Save the resized frame
        frame_filename = os.path.join(output_dir, f"{prefix}_frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame_resized)

        frame_count += 1

    cap.release()
    
# Function to process a single entry in the annotations CSV
def process_video_entry(row, output_base_dir, img_height, img_width):
    video_path = row['rgb_path']
    # annotations_path = row['annotations_path']
    # accident_time = row['accident_time']
    accident_frame = row['accident_frame']
    total_frames = row['no_frames']
    
    # Load annotations (e.g., from a JSON file)
    # with open(annotations_path, 'r') as f:
    #     annotations = json.load(f)

    # Define directories for non-accident and accident frames
    non_accident_dir = os.path.join(output_base_dir, 'Non_Accident')
    accident_dir = os.path.join(output_base_dir, 'Accident')

    # Create necessary directories
    os.makedirs(non_accident_dir, exist_ok=True)
    os.makedirs(accident_dir, exist_ok=True)

    extract_frames(video_path, 0, accident_frame, non_accident_dir, os.path.basename(video_path), img_height, img_width)
    extract_frames(video_path, accident_frame, total_frames, accident_dir, os.path.basename(video_path), img_height, img_width)

# Main function to process the dataset and split it into train, test, and val
def process_dataset(csv_path, output_base_dir, img_height, img_width, test_size=0.2, val_size=0.1):
    df = pd.read_csv(csv_path)

    df_filtered = df[~df['rgb_path'].str.contains('sideswipe', case=False, na=False)]

    # Split into train, test, and validation sets
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)

    # Process videos and create directories
    for i, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="Processing Training Data"):
        process_video_entry(row, os.path.join(output_base_dir, 'train'), img_height, img_width)
    
    for i, row in tqdm(val_data.iterrows(), total=val_data.shape[0], desc="Processing Validation Data"):
        process_video_entry(row, os.path.join(output_base_dir, 'val'), img_height, img_width)
    
    for i, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Processing Testing Data"):
        process_video_entry(row, os.path.join(output_base_dir, 'test'), img_height, img_width)

    print("Dataset processing complete!")

# Run the process with your CSV file and desired image size
csv_path = 'labels.csv'  # Path to your CSV file
output_base_dir = 'data'  # Base directory for train, val, and test
img_height, img_width = 224, 224  # Desired image size

process_dataset(csv_path, output_base_dir, img_height, img_width)