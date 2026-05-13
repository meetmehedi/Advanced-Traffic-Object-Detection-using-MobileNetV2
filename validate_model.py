import os
import glob
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import json

# 1. Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train", "Final Train Dataset")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "traffic_classifier_torch.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# 2. Parse XML
def parse_xml_to_list(xml_dir):
    data = []
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    print(f"Parsing {len(xml_files)} XML files for validation...")
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            src_filename = root.find('filename').text
            img_path = os.path.join(xml_dir, src_filename)
            if not os.path.exists(img_path): continue
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if "CNG" in class_name: class_name = "three wheelers (CNG)"
                
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    
                    if xmax-xmin > 10 and ymax-ymin > 10:
                        data.append({
                            'img_path': img_path,
                            'class': class_name,
                            'bbox': (xmin, ymin, xmax, ymax)
                        })
        except: continue
    return data

annotations = parse_xml_to_list(TRAIN_DIR)

# 3. Label Encoding
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
le = LabelEncoder()
le.classes_ = np.array(class_names)

# Split (Same seed as train_torch.py)
dataset_len = len(annotations)
indices = list(range(dataset_len))
np.random.seed(42)
np.random.shuffle(indices)
train_split = int(0.8 * dataset_len)
val_indices = indices[train_split:]
val_annotations = [annotations[i] for i in val_indices]

# 4. Dataset class
class TrafficDataset(Dataset):
    def __init__(self, annotations, transform=None):
        self.annotations = annotations
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = cv2.imread(ann['img_path'])
        if img is None:
             img = np.zeros((300, 300, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        xmin, ymin, xmax, ymax = ann['bbox']
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        
        crop = img[ymin:ymax, xmin:xmax]
        if crop.size == 0:
             crop = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
        crop = cv2.resize(crop, IMG_SIZE)
        
        if self.transform:
            crop = self.transform(crop)
            
        label = le.transform([ann['class']])[0]
        return crop, label

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = TrafficDataset(val_annotations, transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. Load Model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = models.mobilenet_v2()
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, len(class_names))
)
model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
model = model.to(device)
model.eval()

# 6. Evaluate
all_preds = []
all_labels = []

print("Running validation inference...")
with torch.no_grad():
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        if (i+1) % 50 == 0:
             print(f"Processed batch {i+1}...")

print("\n--- CLASSIFICATION REPORT ---")
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

# Save report
with open("validation_report.txt", "w") as f:
    f.write(report)
print(f"Report saved to validation_report.txt")
