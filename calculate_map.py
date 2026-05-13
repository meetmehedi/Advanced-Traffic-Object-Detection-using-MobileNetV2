import os
import glob
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import numpy as np
import json
from tqdm import tqdm

# 1. Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train", "Final Train Dataset")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "traffic_classifier_torch.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = (128, 128)
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.30

# 2. Load Metadata
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
num_classes = len(class_names)

# 3. Load Models
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
detector = YOLO(YOLO_MODEL_PATH)
classifier = models.mobilenet_v2()
classifier.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(classifier.last_channel, num_classes)
)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
classifier = classifier.to(device)
classifier.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Parse Ground Truth
def parse_xml(xml_dir):
    gt = {}
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            src_filename = root.find('filename').text
            img_path = os.path.join(xml_dir, src_filename)
            if not os.path.exists(img_path): continue
            
            objs = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                if "CNG" in name: name = "three wheelers (CNG)"
                if name not in class_names: continue
                
                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                objs.append({'class': name, 'bbox': [xmin, ymin, xmax, ymax]})
            
            if objs:
                gt[img_path] = objs
        except: continue
    return gt

print("Loading ground truth data...")
all_gt = parse_xml(TRAIN_DIR)
img_paths = list(all_gt.keys())
np.random.seed(42)
np.random.shuffle(img_paths)
val_paths = img_paths[int(0.8 * len(img_paths)):]
print(f"Evaluating on {len(val_paths)} validation images...")

# 5. IoU Function
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

# 6. Evaluation Loop
results = {cls: {"tp": 0, "fp": 0, "total_gt": 0} for cls in class_names}
for path in tqdm(val_paths):
    img = cv2.imread(path)
    if img is None: continue
    h, w = img.shape[:2]
    
    # GT for this image
    img_gt = all_gt[path]
    for obj in img_gt:
        results[obj['class']]["total_gt"] += 1
    
    # Detections
    yolo_results = detector(img, conf=CONF_THRESHOLD, verbose=False)[0]
    preds = []
    
    for box in yolo_results.boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        yolo_cls = int(box.cls[0].item())
        yolo_name = detector.names[yolo_cls]
        
        # We only evaluate classes our classifier knows (or Person from YOLO)
        if yolo_name == "person":
             label = "person" # Assuming person is in class_names if we care
             if label not in class_names: continue
        else:
            crop = img[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
            if crop.size == 0: continue
            input_tensor = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = classifier(input_tensor)
                _, predicted = outputs.max(1)
                label = class_names[predicted.item()]
        
        preds.append({'class': label, 'bbox': [bx1, by1, bx2, by2]})

    # Match Preds to GT
    matched_gt = [False] * len(img_gt)
    for p in preds:
        best_iou = 0
        best_gt_idx = -1
        for i, g in enumerate(img_gt):
            if matched_gt[i]: continue
            if p['class'] == g['class']:
                iou = calculate_iou(p['bbox'], g['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        if best_iou >= IOU_THRESHOLD:
            results[p['class']]["tp"] += 1
            matched_gt[best_gt_idx] = True
        else:
            results[p['class']]["fp"] += 1

# 7. Final Metrics
print("\n--- mAP Evaluation Results (IoU=0.5) ---")
total_precision = []
for cls, stats in results.items():
    if stats['total_gt'] == 0: continue
    precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
    recall = stats['tp'] / stats['total_gt']
    total_precision.append(precision)
    print(f"{cls:25} | Precision: {precision:.4f} | Recall: {recall:.4f}")

mAP = np.mean(total_precision) if total_precision else 0
print(f"\nAggregate mAP@0.5: {mAP:.4f}")
