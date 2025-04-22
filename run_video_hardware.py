import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import time
from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    num_threads = 8  # Set this to the number of threads you want to use
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    cv2.setNumThreads(num_threads)
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str, default="/home/gekot_techlab/Desktop/video.mp4")
    parser.add_argument('--input-size', type=int, default=256)  # Smaller sizes are faster but lower quality
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='Only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='Do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'  # Use MPS if available, otherwise fallback to CPU
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Load and prepare model
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    with torch.no_grad():
        # Load weights and apply dynamic quantization
        depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
        depth_anything = torch.quantization.quantize_dynamic(
            depth_anything, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        depth_anything = torch.compile(depth_anything.eval(),mode = "max-autotune-no-cudagraphs").to(DEVICE)
        print("Model loaded and ready.")
    
    if os.path.isfile(args.video_path):
        filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    start_time = time.time()
    for k, filename in enumerate(filenames):
        print(f'Processing {k + 1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        target_frame_rate = 5  # Process at 5 FPS
        
        # Resize output to match the reduced size
        output_width = (frame_width // 2) if args.pred_only else (frame_width + 50) // 2
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_processed.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), target_frame_rate, (output_width, frame_height // 2))
        
        frames_processed = 0
        frame_interval = max(1, frame_rate // target_frame_rate)
        
        while raw_video.isOpened():
            # Read the next frame
            ret, raw_frame = raw_video.read()
            if not ret or frames_processed >= 20:  # Limit frames for testing, remove this in production
                break
            
            current_frame_number = int(raw_video.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame_number % frame_interval != 0:
                continue
            
            frames_processed += 1
            print(f"Processing frame {frames_processed}")
            
            # Resize the raw frame to half its size
            raw_frame = cv2.resize(raw_frame, (frame_width // 2, frame_height // 2))
            
            # Infer depth using the model
            depth = depth_anything.infer_image(raw_frame, args.input_size)
            depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1]
            
            # Create the output frame
            if args.pred_only:
                out.write(depth)
            else:
                split_region = np.ones((frame_height // 2, 50, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, depth])
                out.write(combined_frame)
        
        raw_video.release()
        out.release()
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
