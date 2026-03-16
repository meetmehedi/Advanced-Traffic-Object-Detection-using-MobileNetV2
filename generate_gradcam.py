import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

# 1. Config
MODEL_PATH = "/Users/md.mehedihasan/Downloads/TRaffic/traffic_classifier_torch.pth"
CLASS_NAMES_PATH = "/Users/md.mehedihasan/Downloads/TRaffic/class_names.json"
DATA_DIR = "/Users/md.mehedihasan/Downloads/TRaffic/train/Final Train Dataset"
OUTPUT_DIR = "/Users/md.mehedihasan/Downloads/TRaffic/detection_outputs/xai"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load Class Names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

# 3. Model with hooks for Grad-CAM
class GradCAMModel(nn.Module):
    def __init__(self, model_path, num_classes):
        super(GradCAMModel, self).__init__()
        self.model = models.mobilenet_v2()
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.model.last_channel, num_classes)
        )
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        # Hook for features[-1] which is the last conv layer
        self.model.features[-1].register_forward_hook(self.save_activation)
        self.model.features[-1].register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.model(x)

    def get_gradcam(self, class_idx):
        # Weight the channels by the average gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU and normalization
        grad_cam = torch.relu(grad_cam)
        grad_cam = grad_cam.detach().cpu().numpy()
        if grad_cam.max() > 0:
            grad_cam = grad_cam / grad_cam.max()
        return grad_cam

# 4. Processing
def process_image(img_path, target_class=None):
    img = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0)
    
    output = model_cam(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    model_cam.model.zero_grad()
    output[0, target_class].backward()
    
    heatmap = model_cam.get_gradcam(target_class)
    
    # Overlay
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (128, 128))
    heatmap_resized = cv2.resize(heatmap, (128, 128))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * 0.4 + img_cv * 0.6
    
    return superimposed_img, class_names[target_class]

# 5. Main Execution
if __name__ == "__main__":
    num_classes = len(class_names)
    model_cam = GradCAMModel(MODEL_PATH, num_classes)
    
    # Select some representative images (hardcoded for now based on previous listings)
    sample_images = [
        "Asraf_01.jpg", # Bus/Car?
        "Asraf_10.jpg",
        "Asraf_20.jpg"
    ]
    
    for filename in sample_images:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            result, label = process_image(path)
            out_path = os.path.join(OUTPUT_DIR, f"gradcam_{filename.replace('.jpg', '')}_{label}.jpg")
            cv2.imwrite(out_path, result)
            print(f"Saved Grad-CAM for {filename} as {label}")
