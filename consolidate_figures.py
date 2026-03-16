import cv2
import os
import matplotlib.pyplot as plt

XAI_DIR = "/Users/md.mehedihasan/Downloads/TRaffic/detection_outputs/xai"
files = [f for f in os.listdir(XAI_DIR) if f.startswith("gradcam_")]

fig, axes = plt.subplots(1, len(files), figsize=(15, 5))
if len(files) == 1: axes = [axes]

for ax, filename in zip(axes, sorted(files)):
    img = cv2.imread(os.path.join(XAI_DIR, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    label = filename.split("_")[-1].replace(".jpg", "")
    ax.set_title(f"Class: {label.capitalize()}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("/Users/md.mehedihasan/Downloads/TRaffic/detection_outputs/xai/figure4_gradcam_consolidated.png")
print("Consolidated Grad-CAM figure saved as figure4_gradcam_consolidated.png")
