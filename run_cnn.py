import os
import glob
import pandas as pd
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train", "Final Train Dataset")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

print("Parsing XML files...")
def parse_xml_to_df(xml_dir):
    data = []
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    for xml_file in xml_files:
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
            img = cv2.imread(img_path) # Changed back to cv2 to avoid any conflicts with PIL
            if img is not None:
                h, w = img.shape[:2]
                ymin = max(0, row['ymin'])
                ymax = min(h, row['ymax'])
                xmin = max(0, row['xmin'])
                xmax = min(w, row['xmax'])
                
                cropped_img = img[ymin:ymax, xmin:xmax]
                if cropped_img.size != 0 and cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                    resized_img = cv2.resize(cropped_img, IMG_SIZE)
                    images.append(resized_img)
                    labels.append(row['class_encoded'])
                    count += 1
        if count % 1000 == 0 and count > 0:
            print(f"Loaded {count} images...")
            
    images = np.array(images, dtype='float32') / 255.0
    labels = to_categorical(np.array(labels), num_classes=num_classes)
    return images, labels

# Process up to 5000 objects to avoid slow execution in agent loop
df_subset = df.sample(frac=1.0, random_state=42).head(5000)
X, y = load_and_preprocess_images(df_subset, TRAIN_DIR)
print(f"Total dataset size for training: {X.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

print("Training model...")
cnn_model = create_cnn_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)
history = cnn_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=1
)

val_loss, val_acc = cnn_model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

y_pred = cnn_model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

report = classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_, zero_division=0)
print("\n--- Classification Report ---")
print(report)

with open(os.path.join(BASE_DIR, 'classification_report.txt'), 'w') as f:
    f.write("Validation Accuracy: {:.2f}%\n\n".format(val_acc*100))
    f.write("--- Classification Report ---\n")
    f.write(report)

# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN Loss')
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
print("Saved training_history.png")

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
print("Saved confusion_matrix.png")
print("Finished!")
