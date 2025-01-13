import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from config import *

# Load and preprocess an image from file path
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    return img

# Prepare test datasets for testing
def prepare_test_datasets(df):
    image_paths = df['filename'].values
    type_labels = tf.keras.utils.to_categorical(df[ATTRIBUTE_TYPE].values, ATTRIBUTE_TYPE_COUNT)
    manufacturer_labels = tf.keras.utils.to_categorical(df[ATTRIBUTE_MANUFACTURER].values, ATTRIBUTE_MANUFACTURER_COUNT)
    bboxes = df[['x1_relative', 'y1_relative', 'x2_relative', 'y2_relative']].values

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, (type_labels, manufacturer_labels, bboxes)))

    dataset = dataset.map(
        lambda x, y: (load_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).map(
        lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset

# Evaluate the classification task for a given attribute
def evaluate_classification(predicted_labels, true_labels, attribute):
    true_labels_indices = np.argmax(true_labels, axis=1)

    accuracy = accuracy_score(true_labels_indices, predicted_labels)
    f1_macro = f1_score(true_labels_indices, predicted_labels, average='macro')
    
    print(f"{attribute} Accuracy: {accuracy:.4f}")
    print(f"{attribute} F1-Score Macro: {f1_macro:.4f}")

# Evaluate the localization task and calculate the mean IoU
def evaluate_localization(predicted_bboxes, true_bboxes):
    iou_scores = []
    for true_box, pred_box in zip(true_bboxes, predicted_bboxes):
        iou = calculate_iou(true_box, pred_box)
        iou_scores.append(iou)

    thresholds = [0.5, 0.75, 0.85, 0.9]
    for thresh in thresholds:
        map = np.mean([iou >= thresh for iou in iou_scores])
        print(f"mAP@{thresh}: {map:.4f}")
    
    mean_iou = np.mean(iou_scores)
    print(f"Detection Mean IoU: {mean_iou:.4f}")

    return mean_iou

# Calculate the Intersection over Union (IoU) for two bounding boxes
def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1_intersect = max(box1[0], box2[0])
    y1_intersect = max(box1[1], box2[1])
    x2_intersect = min(box1[2], box2[2])
    y2_intersect = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2_intersect - x1_intersect) * max(0, y2_intersect - y1_intersect)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def main():
    testing_annotations = pd.read_csv(LABELS_FILENAME_NORMALIZED, sep=";")

    # Load saved models
    single_type_model = tf.keras.models.load_model('type_trained.keras')
    single_manufacturer_model = tf.keras.models.load_model('manufacturer_trained.keras')
    single_localization_model = tf.keras.models.load_model('localization_trained.keras')
    mtl_model = tf.keras.models.load_model('multitask_trained.keras')

    test_ds = prepare_test_datasets(testing_annotations)
    
    # Extract true labels
    type_true = []
    manufacturer_true = []
    localization_true = []

    for _, (type_y, manufacturer_y, localization_y) in test_ds:
        type_true.extend(type_y.numpy())
        manufacturer_true.extend(manufacturer_y.numpy())
        localization_true.extend(localization_y.numpy())

    type_true = np.array(type_true)
    manufacturer_true = np.array(manufacturer_true)
    localization_true = np.array(localization_true)

    # Make predictions
    print("Making predictions for Single-Task Models ...")
    type_pred = single_type_model.predict(test_ds)
    manufacturer_pred = single_manufacturer_model.predict(test_ds)
    localization_pred = single_localization_model.predict(test_ds)
    print("Making predictions for Multi-Task Model ...")
    mtl_pred = mtl_model.predict(test_ds)

    predicted_type_labels = type_pred.argmax(axis=1)
    prd_manufacturer_labels = manufacturer_pred.argmax(axis=1)

    # Evaluate the models
    print("\nSingle-Task Models:")
    evaluate_classification(predicted_type_labels, type_true, ATTRIBUTE_TYPE)
    evaluate_classification(prd_manufacturer_labels, manufacturer_true, ATTRIBUTE_MANUFACTURER)
    evaluate_localization(localization_pred, localization_true)
    print("\nMulti-Task Models:")
    evaluate_classification(mtl_pred[0].argmax(axis=1), type_true, ATTRIBUTE_TYPE)
    evaluate_classification(mtl_pred[1].argmax(axis=1), manufacturer_true, ATTRIBUTE_MANUFACTURER)
    evaluate_localization(mtl_pred[2], localization_true)

if __name__ == "__main__":
    main()