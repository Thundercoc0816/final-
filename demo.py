import os
import random
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))


#------------- Configuration -------------#
# Path to saved model weights (in same folder as this script)
MODEL_PATH = "model.pth"
# Root folder for Animals10 (contains 'train' and 'val')
DATA_ROOT = "Animals10"
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------- Model Setup -------------#
# Load pretrained architecture and replace final layer for 10 classes
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

#------------- Transforms -------------#
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Hardcoded class names matching training order
classes = [
    "Butterfly","Cat","Chicken","Cow","Dog",
    "Elephant","Horse","Sheep","Spider","Squirrel"
]

#------------- Image Gathering -------------#
def collect_image_paths(root_dir):
    paths = []
    for split in ("train", "val"):
        split_dir = os.path.join(root_dir, split)
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(".jpg"):
                    paths.append(os.path.join(cls_dir, fname))
    return paths

#------------- Prediction & Display -------------#
def predict_image(path):
    image = Image.open(path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        prob, idx = torch.softmax(logits, 1).max(1)
    return image, classes[idx.item()], prob.item()

if __name__ == "__main__":
    # Collect all image paths 
    all_images = collect_image_paths(DATA_ROOT)
    sample_paths = random.sample(all_images, 3)

    # Plot each prediction
    for path in sample_paths:
        img, label, score = predict_image(path)
        plt.figure(figsize=(4,4))
        plt.imshow(img)
        plt.title(f"Prediction: {label} ({score*100:.1f}%)")
        plt.axis('off')
    plt.show(block=True)


