import os
import glob
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Config
TRAIN_DIR = "/Users/md.mehedihasan/Downloads/TRaffic/train/Final Train Dataset"
MODEL_SAVE_PATH = "traffic_classifier_torch.pth"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10 

# 2. Parse XML
def parse_xml_to_list(xml_dir):
    data = []
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    print(f"Found {len(xml_files)} XML files.")
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            src_filename = root.find('filename').text
            img_path = os.path.join(xml_dir, src_filename)
            if not os.path.exists(img_path): continue
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                # Normalization
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

print("Parsing annotations...")
annotations = parse_xml_to_list(TRAIN_DIR)
print(f"Total instances extracted: {len(annotations)}")

# 3. Label Encoding
le = LabelEncoder()
classes = [a['class'] for a in annotations]
encoded_labels = le.fit_transform(classes)
for i, a in enumerate(annotations):
    a['label'] = encoded_labels[i]

num_classes = len(le.classes_)
print(f"Classes ({num_classes}): {le.classes_}")
with open(CLASS_NAMES_PATH, 'w') as f:
    json.dump(list(le.classes_), f)

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
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
             crop = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
        
        crop = cv2.resize(crop, IMG_SIZE)
        
        if self.transform:
            crop = self.transform(crop)
            
        return crop, ann['label']

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split
dataset_len = len(annotations)
indices = list(range(dataset_len))
np.random.seed(42)
np.random.shuffle(indices)
train_split = int(0.8 * dataset_len)
train_indices, val_indices = indices[:train_split], indices[train_split:]

train_dataset = TrafficDataset([annotations[i] for i in train_indices], transform=transform_train)
val_dataset = TrafficDataset([annotations[i] for i in val_indices], transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. Model
device = torch.device("cpu") 
print(f"Using device: {device}")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

# 6. Training Loop
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i+1) % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.2f}%")
            running_loss = 0.0
            
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100.*val_correct/val_total
    print(f"Validation Acc after Epoch {epoch+1}: {val_acc:.2f}%")
    scheduler.step(val_acc)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model saved (Acc: {best_acc:.2f}%)")
    
    torch.save(model.state_dict(), f"traffic_classifier_epoch_{epoch+1}.pth")

print(f"Training Finished. Best Acc: {best_acc:.2f}%")
