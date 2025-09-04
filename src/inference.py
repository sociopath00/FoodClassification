import torch
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import torch.nn as nn

from PIL import Image
from pathlib import Path
from typing import Tuple
import config


def efficientnet_inference(img_path: str, 
                           class_names: list) -> Tuple[str, float, torch.Tensor]:
    """Inference function for classifing the image

    Args:
        img_path: Image path for prediction
        class_names: list of output variables

    Returns:
        prediction class
        
    """
    # Read the image
    img = Image.open(img_path).convert("RGB")

    # Fetch transforms from EfficientNet model
    weights = EfficientNet_B1_Weights.DEFAULT
    auto_transforms = weights.transforms()

    # Preprocess/transform
    input_tensor = auto_transforms(img).unsqueeze(0)  # shape: [1, 3, 240, 240]

    # Recreate the base model
    model = efficientnet_b1(weights=None)  # don't load pretrained weights here

    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Recreate classifier head
    output_shape = len(class_names) 
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=output_shape, bias=True)
    )

    MODEL_PATH = Path(config.MODEL_DIR) / config.MODEL_NAME

    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH))

    # Put into eval mode 
    model.eval()

    # Inference
    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]  # convert logits → probabilities
        pred_class_idx = torch.argmax(probabilities).item()
        pred_class = class_names[pred_class_idx]
        pred_confidence = probabilities[pred_class_idx].item()
        # pred_class = torch.argmax(output, dim=1).item()

    return pred_class, round(pred_confidence, 2), probabilities


if __name__ == "__main__":
    img_path = config.TEST_IMG
    class_names = ["pizza", "steak", "sushi"]
    prediction, conf, probs = efficientnet_inference(img_path=img_path, class_names=class_names)
    print(f"Predicted Class: {prediction} with {conf}% confidence")
    print(f"Probabilities are {probs}")

