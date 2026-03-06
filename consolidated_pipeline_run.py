import os
import glob
import pandas as pd
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. Initialize Paths & Hyperparameters
TRAIN_DIR = r'e:\traffic\train\Final Train Dataset' # Correctly updated the data path
IMG_SIZE = (64, 64)
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
                h, w = img.shape[:2]
                ymin, ymax = max(0, row['ymin']), min(h, row['ymax'])
                xmin, xmax = max(0, row['xmin']), min(w, row['xmax'])
                
                cropped_img = img[ymin:ymax, xmin:xmax]
                if cropped_img.size != 0 and cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                    resized_img = cv2.resize(cropped_img, IMG_SIZE)
                    images.append(resized_img)
                    labels.append(row['class_encoded'])
                    count += 1
        if count % 1000 == 0 and count > 0:
            print(f"Loaded {count} images...")
            
    return np.array(images, dtype='float32') / 255.0, to_categorical(np.array(labels), num_classes=num_classes)

# Process a manageable subset to finish training quickly in this demonstration context
df_subset = df.sample(frac=1.0, random_state=42).head(5000) 
X, y = load_and_preprocess_images(df_subset, TRAIN_DIR)
print(f"Dataset successfully loaded. Total Sample Size: {X.shape[0]}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Define CNN Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 7. Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 8. Train the Model
print("Initiating CNN Training...")
history = model.fit(X_train, y_train, epochs=1, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)

# 9. Evaluate CNN Model Performance
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%\n")

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
plt.title('Confusion Matrix - CNN Performance')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Save figures instead of showing them in an un-headed terminal
plt.savefig(r'e:\traffic\consolidated_confusion_matrix.png')
print("Successfully generated and saved unified evaluation plots.")
