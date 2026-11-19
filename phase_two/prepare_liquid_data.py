import cv2
import pandas as pd
import os
from tqdm import tqdm

# Cleanly trimmed segments (cuts off the last ~12-15s of the two CCW runs)
DATASETS = [
    {'prefix': 'driving_20261013_043552', 'segments': [(0, 4100)]}, # Clockwise
    {'prefix': 'driving_20261013_044230', 'segments': [(0, 1000)]}, # CCW 1 (trimmed)
    {'prefix': 'driving_20261013_045230', 'segments': [(0, 2775)]}, # CCW 2 (trimmed)
]

output_dir = "liquid_training_images"
os.makedirs(output_dir, exist_ok=True)

all_rows = []
global_frame_idx = 0
episode_id = 0

print("Unpacking trimmed videos into 'liquid_training_images/'...")

for ds in DATASETS:
    prefix = ds['prefix']
    print(f"\nProcessing {prefix}...")
    
    df = pd.read_csv(f"{prefix}.csv")
    cap = cv2.VideoCapture(f"{prefix}.avi")
    
    for seg_start, seg_end in ds['segments']:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        seg_df = df[(df['frame_id'] >= seg_start) & (df['frame_id'] <= seg_end)]
        
        for _, row in tqdm(seg_df.iterrows(), total=len(seg_df)):
            ret, frame = cap.read()
            if not ret:
                break
                
            resized = cv2.resize(frame, (200, 66))
            img_filename = f"{global_frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, img_filename), resized)
            
            all_rows.append({
                'episode_id': episode_id,
                'image_path': img_filename,
                'steering': row['steering'],
                'throttle': row['throttle'],
                'servo3_pwm': row['servo3_pwm'] if 'servo3_pwm' in row else 1500
            })
            global_frame_idx += 1
            
        episode_id += 1
        
    cap.release()

master_df = pd.DataFrame(all_rows)
master_df.to_csv("liquid_master_labels.csv", index=False)

print(f"\nDone! Extracted {global_frame_idx} clean frames across {episode_id} episodes.")
print("Saved labels to 'liquid_master_labels.csv'.")