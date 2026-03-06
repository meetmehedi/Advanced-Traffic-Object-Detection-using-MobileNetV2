import os
import glob
import pandas as pd
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

TRAIN_DIR = '/Users/md.mehedihasan/Downloads/trafic/train/Final Train Dataset'

IMG_SIZE = (64, 64)

print("Parsing XML files...")
def parse_xml_to_df(xml_dir):
    data = []
    # Avoid glob by using os.listdir and filtering locally
    xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"Found {len(xml_files)} xml files to parse.")
    
    for i, xml_file in enumerate(xml_files):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename_elem = root.find('filename')
            if filename_elem is None:
                continue
            filename = filename_elem.text
            
            size_elem = root.find('size')
            if size_elem is not None:
                 width = int(size_elem.find('width').text)
                 height = int(size_elem.find('height').text)
            else:
                 width, height = 0, 0
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    
                    data.append({
                        'filename': filename,
                        'class': class_name,
                        'width': width,
                        'height': height,
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    })
        except ET.ParseError as e:
            pass # Silently skip malformed
        except Exception as e:
            pass
            
        if (i+1) % 500 == 0:
            print(f"Parsed {i+1}/{len(xml_files)} files...")
            
    return pd.DataFrame(data)

start_time = time.time()
df = parse_xml_to_df(TRAIN_DIR)
print(f"Parsed {len(df)} annotations in {time.time() - start_time:.2f} seconds.")

df['bbox_width'] = df['xmax'] - df['xmin']
df['bbox_height'] = df['ymax'] - df['ymin']

df = df[(df['bbox_width'] > 10) & (df['bbox_height'] > 10)]

label_encoder = LabelEncoder()
df['class_encoded'] = label_encoder.fit_transform(df['class'])
num_classes = len(label_encoder.classes_)
print(f"Classes Found ({num_classes}): {label_encoder.classes_}")

print("Loading and preprocessing images...")
def load_and_preprocess_images(dataframe, img_dir):
    images = []
    labels = []
    
    count = 0
    for _, row in dataframe.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                ymin = max(0, row['ymin'])
                ymax = min(h, row['ymax'])
                xmin = max(0, row['xmin'])
                xmax = min(w, row['xmax'])
                
                cropped_img = img[ymin:ymax, xmin:xmax]
                if cropped_img.size != 0 and cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                    resized_img = cv2.resize(cropped_img, IMG_SIZE)
                    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                    images.append(gray.flatten())
                    labels.append(row['class_encoded'])
                    count += 1
        if count % 1000 == 0 and count > 0:
            print(f"Loaded {count} image crops...")
            
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)
    return images, labels

# Process subset faster: 
df_subset = df.sample(frac=1.0, random_state=42).head(5000)
X, y = load_and_preprocess_images(df_subset, TRAIN_DIR)
print(f"Total dataset size for training: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training RandomForestClassifier model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("Evaluating...")
y_pred = rf.predict(X_test)

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
print("\n--- Classification Report ---")
print(report)

acc = np.mean(y_pred == y_test)
with open('/Users/md.mehedihasan/Downloads/trafic/classification_report.txt', 'w') as f:
    f.write("Validation Accuracy: {:.2f}%\n\n".format(acc*100))
    f.write("--- Classification Report ---\n")
    f.write(report)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - RandomForest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/Users/md.mehedihasan/Downloads/trafic/confusion_matrix.png')
print("Saved confusion_matrix.png")
print("Finished!")
