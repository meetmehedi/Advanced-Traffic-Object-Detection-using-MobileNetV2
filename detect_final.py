import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import json
import numpy as np

# 1. Config
YOLO_MODEL_PATH = "yolov8n.pt"
CLASSIFIER_PATH = "traffic_classifier_torch.pth"
CLASS_NAMES_PATH = "class_names.json"
TEST_DIRS = [
    "/Users/md.mehedihasan/Downloads/TRaffic/test/test",
    "/Users/md.mehedihasan/Downloads/TRaffic/test round 2/test round 2"
]
OUTPUT_DIR = "/Users/md.mehedihasan/Downloads/TRaffic/detection_outputs/final"
IMG_SIZE = (128, 128)
CONF_THRESH = 0.45 # Increased to filter out noisy low-confidence results

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load Class Names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
num_classes = len(class_names)

# 3. Load Models
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# YOLOv8 for Localization
detector = YOLO(YOLO_MODEL_PATH)

# MobileNetV2 for Classification
classifier = models.mobilenet_v2()
classifier.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(classifier.last_channel, num_classes)
)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
classifier = classifier.to(device)
classifier.eval()

# 4. Transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Inference
def run_detection(img_path):
    img = cv2.imread(img_path)
    if img is None: return
    
    results = detector(img, conf=CONF_THRESH, verbose=False)[0]
    annotated = img.copy()
    
    h, w = img.shape[:2]
    
    for box in results.boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Crop and Classify
        crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0: continue
        
        input_tensor = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = classifier(input_tensor)
            _, predicted = outputs.max(1)
            label = class_names[predicted.item()]
            conf = torch.softmax(outputs, dim=1)[0][predicted].item()
            
            # Additional filtering: only use the label if the classifier is also confident
            if conf < 0.35: continue 
            
        # Draw Box and Label
        color = (0, 255, 0) if "CNG" in label or "rickshaw" in label else (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        tag = f"{label} {conf:.2f}"
        cv2.putText(annotated, tag, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    out_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), annotated)
    print(f"Saved {out_name}")

# Process a few images from each dir
for d in TEST_DIRS:
    if not os.path.exists(d): continue
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.jpg')][:15] # Process slightly more
    for f in files:
        run_detection(f)

print(f"Finished! Results in {OUTPUT_DIR}")
