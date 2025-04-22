import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import time
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def process_video(video_path, outdir='./vis_video_depth', encoder='vits'):
    # We set the output resolution to 360p.
    # For standard 720p (1280x720) input, we downscale to 640x360.
    desired_width = 640
    desired_height = 360

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the pre-trained depth model
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder))
    depth_anything = depth_anything.to(DEVICE).eval()
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
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
    
    # Determine list of filenames to process
    if os.path.isfile(video_path):
        if video_path.endswith('txt'):
            with open(video_path, 'r') as f:
                lines = f.read().splitlines()
            filenames = lines
        else:
            filenames = [video_path]
    else:
        filenames = os.listdir(video_path)
        filenames = [os.path.join(video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(outdir, exist_ok=True)
    
    # Process each video file
    for k, filename in enumerate(filenames):
        print('Progress {}/{} Processing {}'.format(k+1, len(filenames), filename))
        
        raw_video = cv2.VideoCapture(filename)
        # Note: original dimensions are not used since we are resizing to 360p.
        original_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        
        basename = os.path.basename(filename)
        output_path = os.path.join(outdir, basename[:basename.rfind('.')] + '_video_depth.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (desired_width, desired_height))
        
        prev_depth = None
        
        # Adjusted settings for drawing the trapezoid (projected vehicle path)
        # Bottom edge spans the full width.
        bottom_x1 = 0
        bottom_x2 = desired_width
        frame_center_x = desired_width // 2
        # Top edge spans 50% of the frame width.
        top_width = int(desired_width * 0.5)
        top_x1 = frame_center_x - top_width // 2
        top_x2 = frame_center_x + top_width // 2
        # Set the top y-coordinate at 65% of the frame height.
        top_y = int(desired_height * 0.65)
        danger_threshold = 70
        depth_threshold = 30
        
        temp = time.time()
        
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            # Resize the frame to 360p
            raw_frame = cv2.resize(raw_frame, (desired_width, desired_height))
            
            # Preprocess the frame for depth estimation
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            frame = transform({'image': frame})['image']
            frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                depth = depth_anything(frame)
            
            # Resize depth map to match the 360p resolution
            depth = F.interpolate(depth[None], (desired_height, desired_width), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            
            if prev_depth is not None:
                # Compute depth gradient and amplify differences
                depth_gradient = cv2.absdiff(depth, prev_depth) * 5
                mask = depth_gradient < depth_threshold
                mask = np.stack([mask]*3, axis=-1)
                
                depth_gradient_color = cv2.applyColorMap(depth_gradient, cv2.COLORMAP_COOL)
                depth_masked = np.where(mask, raw_frame, depth_gradient_color)
                
                # Draw trapezoid indicating the projected vehicle path
                trapezoid_bbox = np.array([[
                    (bottom_x1, desired_height), 
                    (top_x1, top_y), 
                    (top_x2, top_y), 
                    (bottom_x2, desired_height)
                ]], dtype=np.int32)
                cv2.drawContours(depth_masked, trapezoid_bbox, -1, (0, 255, 0))
                
                # Check for danger based on depth gradient within the trapezoid region
                if np.any(depth_gradient[top_y:desired_height, top_x1:top_x2] > danger_threshold):
                    cv2.putText(depth_masked, "Danger", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                out.write(depth_masked)
            
            prev_depth = depth
        
        print("Processing time for {}: {:.2f} seconds".format(filename, time.time() - temp))
        raw_video.release()
        out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True,
                        help="Path to a video file, a directory of videos, or a text file with video paths")
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    args = parser.parse_args()
    process_video(args.video_path, args.outdir)
