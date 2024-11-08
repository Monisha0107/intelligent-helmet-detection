import os
import cv2
from pathlib import Path

# Paths for source and destination folders
src_folder = 'dataset'
output_folder = 'processed_dataset'
image_size = (224, 224)  

# Create processed folders
Path(f"{output_folder}/with_helmet").mkdir(parents=True, exist_ok=True)
Path(f"{output_folder}/without_helmet").mkdir(parents=True, exist_ok=True)

# Function to preprocess images
def preprocess_images(category):
    src_path = os.path.join(src_folder, category)
    dest_path = os.path.join(output_folder, category)
    
    # Process each image
    for count, img_name in enumerate(os.listdir(src_path)):
        img_path = os.path.join(src_path, img_name)
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            continue 
        
        # Resize the image
        img_resized = cv2.resize(img, image_size)
        img_normalized = img_resized / 255.0
        
        # Save preprocessed image
        new_filename = f"{category}_{count:03}.jpg"
        new_file_path = os.path.join(dest_path, new_filename)
        cv2.imwrite(new_file_path, (img_normalized * 255).astype('uint8'))

    print(f"Preprocessed {category} images saved in {dest_path}")

# Run preprocessing for both categories
preprocess_images("with_helmet")
preprocess_images("without_helmet")
