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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    
    args = parser.parse_args()
    
    margin_width = 50

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
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

    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = os.listdir(args.video_path)
        filenames = [os.path.join(args.video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 + margin_width
        
        filename = os.path.basename(filename)
        output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + '_video_depth.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))

        prev_depth = None
        
        bottom_width_ratio = 1/3  # Bottom width as a ratio of frame width
        top_width_ratio = 1/6     # Top width as a ratio of frame width
        speed = 10                # Example speed variable, adjust as needed
        danger_threshold = 70     # Example danger threshold, adjust as needed

        temp = time.time()
        frame_center_x = frame_width // 2
        bottom_width = int(frame_width * bottom_width_ratio)
        top_width = int(frame_width * top_width_ratio)
        top_x1 = frame_center_x - top_width // 2
        top_x2 = frame_center_x + top_width // 2
        bottom_x1 = frame_center_x - bottom_width // 2
        bottom_x2 = frame_center_x + bottom_width // 2
        top_y = int(frame_height - speed*10)  # Adjust top y-coordinate based on speed
        depth_threshold = 30

        speed = 10

        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            
            frame = transform({'image': frame})['image']
            frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                depth = depth_anything(frame)

            depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            
            depth = depth.cpu().numpy().astype(np.uint8)
            
            if prev_depth is not None:
                depth_gradient = cv2.absdiff(depth, prev_depth)*5
                depth_original = depth_gradient.copy()

                # Create a mask where depth values are significant
                mask = depth_gradient < depth_threshold
                mask = np.stack([mask]*3, axis=-1)  # Make the mask 3-channel

                # Create a filtered color depth map for visualization
                depth_gradient_color = cv2.applyColorMap(depth_gradient, cv2.COLORMAP_COOL)
                depth_masked = np.where(mask, raw_frame, depth_gradient_color)

                trapezoid_bbox = np.array([[(bottom_x1, frame_height), (top_x1, top_y), (top_x2, top_y), (bottom_x2, frame_height)]], dtype=np.int32)
                cv2.drawContours(depth_masked, trapezoid_bbox, -1, (0, 255, 0))
                if np.any(depth_gradient[top_y:, bottom_x1:bottom_x2] > danger_threshold):
                    cv2.putText(depth_masked, "Danger", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                out.write(depth_masked)
            
            prev_depth = depth
        print(time.time()-temp)
        
        raw_video.release()
        out.release()
