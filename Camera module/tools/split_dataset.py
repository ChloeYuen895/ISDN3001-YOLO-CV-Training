import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
input_img_dir = 'Camera module/dataset/images'  # Temporary folder with all images
input_label_dir = 'Camera module/dataset/labels'  # Optional folder with .txt label files (if already annotated)
output_dir = 'Camera module/dataset'  # Output dataset folder
train_img_dir = os.path.join(output_dir, 'images', 'train')
val_img_dir = os.path.join(output_dir, 'images', 'val')
test_img_dir = os.path.join(output_dir, 'images', 'test')
train_label_dir = os.path.join(output_dir, 'labels', 'train')
val_label_dir = os.path.join(output_dir, 'labels', 'val')
test_label_dir = os.path.join(output_dir, 'labels', 'test')

# Create output directories
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# Get list of image files
images = [f for f in os.listdir(input_img_dir) if f.endswith(('.jpg', '.png'))]

# Split into train (70%), val (15%), and test (15%)
train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

# Move images and corresponding labels
for img in train_imgs:
    # Move image
    shutil.move(os.path.join(input_img_dir, img), os.path.join(train_img_dir, img))
    # Move label if it exists
    label = img.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(input_label_dir, label)
    if os.path.exists(label_path):
        shutil.move(label_path, os.path.join(train_label_dir, label))
    print(f'Moved {img} to train')

for img in val_imgs:
    # Move image
    shutil.move(os.path.join(input_img_dir, img), os.path.join(val_img_dir, img))
    # Move label if it exists
    label = img.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(input_label_dir, label)
    if os.path.exists(label_path):
        shutil.move(label_path, os.path.join(val_label_dir, label))
    print(f'Moved {img} to val')

for img in test_imgs:
    # Move image
    shutil.move(os.path.join(input_img_dir, img), os.path.join(test_img_dir, img))
    # Move label if it exists
    label = img.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(input_label_dir, label)
    if os.path.exists(label_path):
        shutil.move(label_path, os.path.join(test_label_dir, label))
    print(f'Moved {img} to test')

print("Dataset split complete!")
print(f"Train images: {len(train_imgs)}")
print(f"Validation images: {len(val_imgs)}")
print(f"Test images: {len(test_imgs)}")