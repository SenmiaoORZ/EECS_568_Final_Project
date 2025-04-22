import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def load_depth_model(encoder='vits', device='cuda'):
    model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder))
    model = model.to(device).eval()
    return model

def process_video_for_danger(video_file, model, transform, device, desired_width=640, desired_height=360,
                               danger_threshold=70, depth_threshold=30):
    """
    Process a single video file and return 1 if a danger event is detected,
    otherwise 0.
    
    Danger is detected if, after computing the depth gradient between consecutive frames,
    any pixel in the region corresponding to the projected vehicle path exceeds the danger_threshold.
    
    The region is defined as the rectangle from y=top_y (65% of the frame height) to the bottom,
    and spanning 50% of the width (centered horizontally).
    """
    cap = cv2.VideoCapture(video_file)
    danger_detected = False
    prev_depth = None

    # Define region of interest: bottom edge spans full width,
    # top edge spans 50% of the width (centered horizontally) at 65% of frame height.
    bottom_x1 = 0
    bottom_x2 = desired_width
    frame_center_x = desired_width // 2
    top_width = int(desired_width * 0.5)
    top_x1 = frame_center_x - top_width // 2
    top_x2 = frame_center_x + top_width // 2
    top_y = int(desired_height * 0.65)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 360p (assumes 720p input)
        frame = cv2.resize(frame, (desired_width, desired_height))
        # Preprocess the frame for the depth model
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        input_frame = transform({'image': input_frame})['image']
        input_frame = torch.from_numpy(input_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            depth = model(input_frame)

        # Resize the depth map to the desired output resolution
        depth = torch.nn.functional.interpolate(depth[None], (desired_height, desired_width),
                                                mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)

        if prev_depth is not None:
            # Compute the depth gradient between consecutive frames and amplify differences.
            depth_gradient = cv2.absdiff(depth, prev_depth) * 5
            # Extract region corresponding to the projected vehicle path.
            region = depth_gradient[top_y:desired_height, top_x1:top_x2]
            if np.any(region > danger_threshold):
                danger_detected = True
                break

        prev_depth = depth

    cap.release()
    return int(danger_detected)

def run_pipeline(csv_file, video_dir, encoder='vits', desired_width=640, desired_height=360):
    """
    Loads the CSV file and, for each row, processes the corresponding video file
    (only those with file names starting with "converted_") if it exists. The prediction
    (1 if a danger event is detected, 0 otherwise) is logged into the "target" column.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_depth_model(encoder, device)
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # Load the CSV file
    df = pd.read_csv(csv_file)

    # For each row, construct the video filename: "converted_{id}.mp4"
    for idx, row in df.iterrows():
        video_id = row['id']
        video_file = os.path.join(video_dir, f"converted_{video_id}.mp4")
        if not os.path.exists(video_file):
            print(f"Video file {video_file} not found. Skipping.")
            continue
        print(f"Processing video {video_file} ...")
        pred = process_video_for_danger(video_file, model, transform, device,
                                        desired_width, desired_height)
        df.at[idx, 'target'] = pred

    # Save the updated CSV with predictions
    output_csv = os.path.splitext(csv_file)[0] + "_with_predictions.csv"
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', type=str, default='Train.csv', 
                        help='Path to the CSV file with video ids')
    parser.add_argument('--video-dir', type=str, default='.', 
                        help='Directory containing the video files (only files starting with "converted_" are used)')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    
    args = parser.parse_args()
    run_pipeline(args.csv_file, args.video_dir, args.encoder)
