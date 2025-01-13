import os
# Configuration constants
# Classification attributes
ATTRIBUTE_MANUFACTURER = 'manufacturer'
ATTRIBUTE_MANUFACTURER_COUNT = 49

ATTRIBUTE_TYPE = 'type'
ATTRIBUTE_TYPE_COUNT = 9

# Model training parameters
IMAGE_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

# Dataset directories 
DATA_DIR = 'dataset'

DATA_DIR_TRAINING = os.path.join(DATA_DIR, 'train', 'cars_train')
DATA_DIR_TESTING = os.path.join(DATA_DIR, 'test', 'cars_test')

DATA_DIR_TRAINING_NORMALIZED = os.path.join(DATA_DIR, 'train', 'cars_train_normalized')
DATA_DIR_TESTING_NORMALIZED = os.path.join(DATA_DIR, 'test', 'cars_test_normalized')

# Dataset file path
LABELS_FILENAME = os.path.join(DATA_DIR, 'labels', 'train_labels.csv')
LABELS_TEST_FILENAME = os.path.join(DATA_DIR, 'labels', 'test_labels.csv')

LABELS_FILENAME_NORMALIZED = os.path.join(DATA_DIR, 'labels', 'train_labels_normalized.csv')
LABELS_TEST_FILENAME_NORMALIZED = os.path.join(DATA_DIR, 'labels', 'test_labels_normalized.csv')
