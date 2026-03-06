import os
import glob
import pandas as pd
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. Initialize Paths & Hyperparameters
TRAIN_DIR = r'e:\traffic\train\Final Train Dataset' 
IMG_SIZE = (128, 128) # Increased resolution for better feature extraction
BATCH_SIZE = 32
EPOCHS = 10 

# 2. Define XML Parsing Function
def parse_xml_to_df(xml_dir):
    data = []
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename_elem = root.find('filename')
            if filename_elem is None: continue
            filename = filename_elem.text
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    
                    data.append({
                        'filename': filename, 'class': class_name,
                        'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax
                    })
        except ET.ParseError:
            print(f"Skipping badly formatted XML file: {xml_file}")
            continue
    return pd.DataFrame(data)

# 3. Create Annotations DataFrame & Perform Feature Engineering
print("Parsing XML and constructing DataFrame...")
df = parse_xml_to_df(TRAIN_DIR)

if len(df) == 0:
    print("No annotations found. Please check your data directory.")
    exit(1)

df['bbox_width'] = df['xmax'] - df['xmin']
df['bbox_height'] = df['ymax'] - df['ymin']

# Filter out erroneous bounding boxes
df = df[(df['bbox_width'] > 10) & (df['bbox_height'] > 10)]

# 4. Label Encoding
label_encoder = LabelEncoder()
df['class_encoded'] = label_encoder.fit_transform(df['class'])
num_classes = len(label_encoder.classes_)
print(f"Classes Found ({num_classes}): {label_encoder.classes_}")

# 5. Preprocess Images into NumPy Arrays
def load_and_preprocess_images(dataframe, img_dir):
    images, labels = [], []
    count = 0
    for _, row in dataframe.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # Convert BGR to RGB since MobileNetV2 expects RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                ymin, ymax = max(0, row['ymin']), min(h, row['ymax'])
                xmin, xmax = max(0, row['xmin']), min(w, row['xmax'])
                
                cropped_img = img[ymin:ymax, xmin:xmax]
                if cropped_img.size != 0 and cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                    resized_img = cv2.resize(cropped_img, IMG_SIZE)
                    images.append(resized_img)
                    labels.append(row['class_encoded'])
                    count += 1
        if count % 2000 == 0 and count > 0:
            print(f"Loaded {count} images...")
            
    # MobileNet expects inputs in [0, 1] or [-1, 1] depending on preprocess_input, dividing by 255.0 is sufficient mapping
    return np.array(images, dtype='float32') / 255.0, to_categorical(np.array(labels), num_classes=num_classes)

# Process the ENTIRE clean dataset
df_shuffled = df.sample(frac=1.0, random_state=42)
X, y = load_and_preprocess_images(df_shuffled, TRAIN_DIR)
print(f"Dataset successfully loaded. Total Sample Size: {X.shape[0]}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Define Advanced CNN Architecture via Transfer Learning (MobileNetV2)
print("Constructing MobileNetV2 Transfer Learning Model...")
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
])

base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze base model layers initially

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

# 7. Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 8. Train the Model (Head Only)
print(f"Initiating CNN Training for {EPOCHS} epochs...")
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)

# Fine-tuning (Optional, unfreezing top layers of MobileNetV2)
print("Unfreezing top layers of MobileNetV2 for fine-tuning...")
base_model.trainable = True
# Fine-tune from this layer onwards
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate for fine tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print("Fine-tuning model for additional 5 epochs...")
history_fine = model.fit(X_train, y_train, epochs=5, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)

# 9. Evaluate CNN Model Performance
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy after Fine-tuning: {accuracy*100:.2f}%\n")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print Classification Report
report = classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_, labels=np.arange(len(label_encoder.classes_)), zero_division=0)
print("--- Classification Report ---")
print(report)

# Plot Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - Improved CNN (Acc: {accuracy*100:.2f}%)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

plt.savefig(r'e:\traffic\improved_confusion_matrix.png')
print("Successfully generated and saved improved unified evaluation plots.")
