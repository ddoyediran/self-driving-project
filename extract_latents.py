import torch
import cv2
import pandas as pd
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# --- CONFIGURATION ---
# Watch your videos! Find the exact frame numbers where you are driving cleanly.
# If you picked the car up between frames 2000 and 2200, exclude them!
# --- CONFIGURATION ---
DATASETS = [
    {
        'prefix': 'driving_dataset_001',
        # Drive normally, recover from off-track, cut right before the crash
        'segments': [(0, 3285)] 
    },
    {
        'prefix': 'driving_dataset_0012',
        # Drive 1, human picks up car (skip frames 2431-2564), Drive 2
        'segments': [(0, 2430), (2565, 5336)] 
    }
]

print("Loading DINOv2 from Meta...")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
dinov2.eval() 

# Transform: Resize the cropped 480x480 square down to 224x224 and normalize
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

all_latents = []
all_actions = []
all_imu = []
all_terminals = []

for ds in DATASETS:
    prefix = ds['prefix']
    
    print(f"\n--- Processing {prefix} ---")
    df = pd.read_csv(f"{prefix}.csv")
    cap = cv2.VideoCapture(f"{prefix}.avi")
    
    for seg_start, seg_end in ds['segments']:
        print(f"  -> Extracting Segment: Frames {seg_start} to {seg_end}")
        
        latents = []
        # Get actions and IMU for just this segment
        seg_df = df[(df['frame_id'] >= seg_start) & (df['frame_id'] <= seg_end)]
        actions = seg_df[['steering', 'throttle']].values
        imu = seg_df[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
        
        # Set video to the start frame of this segment
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        
        # Extract frames
        with torch.no_grad():
            for _ in tqdm(range(len(seg_df))):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 1. CROP THE IMAGE (Remove left/right black void to make it square)
                # Original is 640x480. We take Y: 0->480, X: 80->560 (a 480x480 square)
                cropped_frame = frame[:, 80:560]
                
                # 2. Convert to RGB for PyTorch
                frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                
                # 3. Transform and send to M1 GPU
                img_tensor = transform(frame_rgb).unsqueeze(0).to(device)
                
                # 4. Extract latent vector
                latent_vector = dinov2(img_tensor)
                latents.append(latent_vector.cpu().numpy().squeeze())
        
        # Create terminals array (Marks the end of this specific segment)
        terminals = np.zeros(len(latents), dtype=bool)
        if len(terminals) > 0:
            terminals[-1] = True 
            
        all_latents.append(np.array(latents))
        all_actions.append(actions)
        all_imu.append(imu)
        all_terminals.append(terminals)

    cap.release()

print("\n--- COMBINING ALL DATA ---")
final_latents = np.concatenate(all_latents, axis=0)
final_actions = np.concatenate(all_actions, axis=0)
final_imu = np.concatenate(all_imu, axis=0)
final_terminals = np.concatenate(all_terminals, axis=0)

print(f"Total Valid Frames: {final_latents.shape[0]}")
print(f"Latent Shape: {final_latents.shape}")

print("Saving to 'training_data.npz'...")
np.savez('training_data.npz', 
         latents=final_latents, 
         actions=final_actions, 
         imu=final_imu,
         terminals=final_terminals)
         
print("Done! You have successfully distilled the world knowledge.")