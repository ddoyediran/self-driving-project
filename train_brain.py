import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- 1. THE NEURAL NETWORK ARCHITECTURE ---
class LatentWorldModel(nn.Module):
    def __init__(self):
        super(LatentWorldModel, self).__init__()
        
        # THE ACTOR: Maps 384 DINOv2 features -> 2 Actions (Steering, Throttle)
        self.actor = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2) 
        )
        
        # THE DYNAMICS MODEL: Maps (384 Features + 2 Actions) -> 384 Next Features
        self.dynamics = nn.Sequential(
            nn.Linear(384 + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 384)
        )

    def forward_actor(self, z_t):
        return self.actor(z_t)

    def forward_dynamics(self, z_t, a_t):
        # Combine state and action into one array
        combined = torch.cat((z_t, a_t), dim=1)
        return self.dynamics(combined)

# --- 2. DATASET FORMATTER ---
# We must carefully format the data so $z_t$ maps to $z_{t+1}$
class DrivingDataset(Dataset):
    def __init__(self, npz_path):
        print("Loading data into memory...")
        data = np.load(npz_path)
        
        self.latents = torch.FloatTensor(data['latents'])
        self.actions = torch.FloatTensor(data['actions'])
        self.terminals = data['terminals']
        
        self.valid_indices = []
        
        # We cannot predict the future if the current frame is the end of a video!
        for i in range(len(self.latents) - 1):
            if not self.terminals[i]: 
                self.valid_indices.append(i)
                
        print(f"Created {len(self.valid_indices)} valid (State -> Next State) pairs.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        z_t = self.latents[real_idx]
        a_t = self.actions[real_idx]
        z_t_next = self.latents[real_idx + 1] # The future!
        
        return z_t, a_t, z_t_next

# --- 3. TRAINING LOOP ---
def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load dataset
    dataset = DrivingDataset('training_data.npz')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = LatentWorldModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # We use Mean Squared Error (MSE) to measure how wrong the predictions are
    mse_loss = nn.MSELoss()
    
    epochs = 30
    
    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        total_actor_loss = 0
        total_dyn_loss = 0
        
        for z_t, a_t, z_t_next in dataloader:
            z_t = z_t.to(device)
            a_t = a_t.to(device)
            z_t_next = z_t_next.to(device)
            
            optimizer.zero_grad()
            
            # 1. Train the Actor (Did it guess the human's steering/throttle?)
            pred_action = model.forward_actor(z_t)
            actor_loss = mse_loss(pred_action, a_t)
            
            # 2. Train the Dynamics (Did it guess the physics of the next frame?)
            pred_z_t_next = model.forward_dynamics(z_t, a_t)
            dyn_loss = mse_loss(pred_z_t_next, z_t_next)
            
            # Combine the losses (We weight the dynamics slightly lower so it prioritizes driving)
            loss = actor_loss + (0.5 * dyn_loss)
            
            loss.backward()
            optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_dyn_loss += dyn_loss.item()
            
        avg_actor = total_actor_loss / len(dataloader)
        avg_dyn = total_dyn_loss / len(dataloader)
        
        print(f"Epoch [{epoch}/{epochs}] | Actor Loss: {avg_actor:.4f} | Dynamics Loss: {avg_dyn:.4f}")

    print("\nTraining Complete!")
    
    # Save the brains!
    torch.save(model.state_dict(), 'latent_world_model.pth')
    print("Saved trained weights to 'latent_world_model.pth'")

if __name__ == '__main__':
    train()