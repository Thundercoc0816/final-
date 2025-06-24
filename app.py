import gradio as gr
import os, random, torch
from torchvision import models, transforms
from PIL import Image

DATA_ROOT  = "Animals10"
MODEL_PATH = "model.pth"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
classes = ["Butterfly","Cat","Chicken","Cow","Dog",
           "Elephant","Horse","Sheep","Spider","Squirrel"]

def predict_upload(img):
    # img: PIL.Image from Gradio
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    prob, idx = torch.softmax(logits,1).max(1)
    return {classes[idx]: float(prob)}


iface = gr.Interface(
    fn=predict_upload,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),
    title="Animals-10 Classifier",
    description="predic ani class"
)

if __name__=="__main__":

    iface.launch(share=True)

