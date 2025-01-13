import pandas as pd
import cv2
import os
from config import *

# Normalize the dataset by resizing the images to a fixed size and updating bounding box coordinates
def main(isTraining=True):
    print("Preprocessing dataset ...")
    labels_filename = LABELS_FILENAME
    labels_filename_normalized = LABELS_FILENAME_NORMALIZED
    data_dir = DATA_DIR_TRAINING
    data_dir_normalized = DATA_DIR_TRAINING_NORMALIZED

    if not isTraining:
        labels_filename = LABELS_TEST_FILENAME
        labels_filename_normalized = LABELS_TEST_FILENAME_NORMALIZED
        data_dir = DATA_DIR_TESTING
        data_dir_normalized = DATA_DIR_TESTING_NORMALIZED

    training_annotations = pd.read_csv(labels_filename, sep=";")

    if not os.path.exists(data_dir_normalized):
        os.makedirs(data_dir_normalized)

    # Add columns for normalized bounding box coordinates
    for col in ['x1_relative', 'y1_relative', 'x2_relative', 'y2_relative', 'filename']:
        if col not in training_annotations.columns:
            training_annotations[col] = None

    # Normalize the dataset by resizing the images to a fixed size and updating bounding box coordinates
    for index, car in training_annotations.iterrows():
        source_filename = os.path.join(data_dir, car['image'])
        source_img = cv2.imread(source_filename)
        height, width, _ = source_img.shape
        
        normalized_bbox = {
            'x1_relative': car['x1'] / width,
            'y1_relative': car['y1'] / height,
            'x2_relative': car['x2'] / width,
            'y2_relative': car['y2'] / height,
        }

        resized_img = cv2.resize(source_img, (IMAGE_SIZE, IMAGE_SIZE))

        target_filename = os.path.join(data_dir_normalized, car['image'])
        cv2.imwrite(target_filename, resized_img)
        
        training_annotations.at[index, 'filename'] = target_filename
        for key, value in normalized_bbox.items():
            training_annotations.at[index, key] = value

    # Save the normalized annotations to a new CSV file
    training_annotations.to_csv(labels_filename_normalized, index=False)

    print(f"Normalized annotations saved to: {labels_filename_normalized}")

if __name__ == "__main__":
    main(isTraining=True)
    main(isTraining=False)
