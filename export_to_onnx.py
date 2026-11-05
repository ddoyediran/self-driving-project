import torch
import torch.nn as nn
from torchvision import transforms

# 1. Recreate the exact architecture so we can load your weights
class LatentWorldModel(nn.Module):
    def __init__(self):
        super(LatentWorldModel, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(384, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.dynamics = nn.Sequential(
            nn.Linear(384 + 2, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 384)
        )
    def forward_actor(self, z_t):
        return self.actor(z_t)

print("Loading trained PyTorch models...")

# Load DINOv2
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cpu()
dinov2.eval()

# Load Your Trained Brain
brain = LatentWorldModel().cpu()
brain.load_state_dict(torch.load('latent_world_model.pth', map_location='cpu'))
brain.eval()

# We only need the Actor network for deployment (to drive the car)
actor_network = brain.actor

# --- EXPORT TO ONNX ---
print("\nExporting DINOv2 to ONNX (This takes a moment)...")
# Create a dummy image tensor matching the exact shape DINOv2 expects: (Batch, Channels, Height, Width)
dummy_image = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    dinov2, dummy_image, "dinov2_encoder.onnx", 
    export_params=True, opset_version=14, 
    do_constant_folding=True,
    input_names=['input_image'], output_names=['latent_vector']
)

print("Exporting Actor Policy to ONNX...")
# Create a dummy latent vector matching the shape the Actor expects: (Batch, Features)
dummy_latent = torch.randn(1, 384)
torch.onnx.export(
    actor_network, dummy_latent, "actor_policy.onnx", 
    export_params=True, opset_version=14, 
    do_constant_folding=True,
    input_names=['latent_vector'], output_names=['action']
)

print("\nSuccess! You have generated the compiled edge models:")
print("1. dinov2_encoder.onnx")
print("2. actor_policy.onnx")
print("You are ready to transfer these back to the Raspberry Pi 5!")