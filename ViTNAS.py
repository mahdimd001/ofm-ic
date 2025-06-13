import torch

# Load the entire model
model = torch.load('ckpt_360.pth')

# Set to eval mode if needed
model.eval()