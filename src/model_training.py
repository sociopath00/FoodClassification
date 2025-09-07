import os
from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

from src.data_setup import create_dataloaders
from src.engine import train
import config

# # Set the manual seeds
# torch.manual_seed(42)
# torch.mps.manual_seed(42)

# Setup device agnostic code
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

# Prepare data paths
image_path = Path(config.DATA_DIR) / config.TRAIN_DATA_PATH

train_dir = image_path / "train"
test_dir = image_path / "test"

# Fetch transforms from EfficientNet model
weights = EfficientNet_B1_Weights.DEFAULT
auto_transforms = weights.transforms()

# Prepare Dataloaders
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=auto_transforms,
    batch_size=32
)

# Setup the model with pretrained weights
model = efficientnet_b1(weights=weights).to(device)

# Freeze all the base layers for pretraining
for param in model.features.parameters():
    param.requires_grad = False

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Setup training and save the results
results = train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=5,
                device=device,
                verbose=1)

# print(f"Model Result: {results}")

# Save the model
MODEL_DIR = Path(config.MODEL_DIR)
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

MODEL_SAVE_PATH = MODEL_DIR / config.MODEL_NAME
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)


