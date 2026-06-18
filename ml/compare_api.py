import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from ml.flower_names import FLOWER_NAMES

MODEL_PATH = Path(__file__).resolve().parent / "models" / "plant_transfer_finetuned_best.pth"

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

_pretrained_model = None
_finetuned_model = None
_class_names = None


def load_pretrained():
    global _pretrained_model

    if _pretrained_model is not None:
        return _pretrained_model

    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    _pretrained_model = (model, weights.meta["categories"])
    return _pretrained_model


def load_finetuned():
    global _finetuned_model, _class_names

    if _finetuned_model is not None:
        return _finetuned_model, _class_names

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    _class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.30),
        nn.Linear(in_features, len(_class_names))
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _finetuned_model = model
    return _finetuned_model, _class_names


def compare_image_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = _transform(image).unsqueeze(0)

    pretrained_model, imagenet_categories = load_pretrained()
    finetuned_model, class_names = load_finetuned()

    with torch.no_grad():
        pre_outputs = pretrained_model(image_tensor)
        pre_probs = torch.softmax(pre_outputs, dim=1)[0]
        pre_conf, pre_idx = torch.topk(pre_probs, 5)

        fine_outputs = finetuned_model(image_tensor)
        fine_probs = torch.softmax(fine_outputs, dim=1)[0]
        fine_conf, fine_idx = torch.topk(fine_probs, 5)

    pretrained_results = []
    for i in range(5):
        idx = int(pre_idx[i].item())
        pretrained_results.append({
            "label": imagenet_categories[idx],
            "confidence": round(float(pre_conf[i].item()), 4)
        })

    finetuned_results = []
    for i in range(5):
        idx = int(fine_idx[i].item())
        class_id = int(class_names[idx].split("_")[1]) + 1
        eng, tr = FLOWER_NAMES.get(class_id, ("unknown", "bilinmeyen"))

        finetuned_results.append({
            "label": f"{eng} ({tr})",
            "confidence": round(float(fine_conf[i].item()), 4)
        })

    return {
        "pretrained_resnet18": pretrained_results,
        "finetuned_resnet18": finetuned_results
    }