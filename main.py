import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
from config import *

# Load and preprocess an image from file path
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    return img

# Prepare an image for training and apply data augmentation
def prepare_image(image, label, isTraining=True):
    if isTraining:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    image = tf.keras.layers.Rescaling(1./255)(image) # Normalize the pixel values
    return image, label

# Prepare a dataset for tasks
def prepare_dataset(df, attribute=False, isTraining=True, isMultiTask=False):
    image_paths = df['filename'].values
    type_labels = tf.keras.utils.to_categorical(df[ATTRIBUTE_TYPE].values, ATTRIBUTE_TYPE_COUNT)
    manufacturer_labels = tf.keras.utils.to_categorical(df[ATTRIBUTE_MANUFACTURER].values, ATTRIBUTE_MANUFACTURER_COUNT)
    bboxes = df[['x1_relative', 'y1_relative', 'x2_relative', 'y2_relative']].values  
    
    # Determine the data labels based on the task
    if isMultiTask:
        data_labels = (type_labels, manufacturer_labels, bboxes)
    elif attribute == ATTRIBUTE_TYPE:
        data_labels = type_labels
    elif attribute == ATTRIBUTE_MANUFACTURER:
        data_labels =  manufacturer_labels
    else:
        data_labels = bboxes

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, data_labels))
    dataset = dataset.map(
        lambda x, y: (load_image(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).cache()
    
    if isTraining:
        dataset = dataset.shuffle(2000) # Shuffle the dataset for training
    
    dataset = dataset.map(
        lambda x, y: prepare_image(x, y, isTraining),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return dataset
def prepare_base_model():
    # Use VGG16 as the base model and feature extractor
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    
    # Set all layers as trainable
    for layer in base_model.layers:
        layer.trainable = True
    return base_model


# Prepare specific layers for classification tasks
def prepare_classification_head(base_model_output, num_classes, name):
    x = GlobalAveragePooling2D()(base_model_output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(num_classes, name=name, activation='softmax')(x)
    return x

# Prepare specific layers for localization task
def prepare_localization_head(base_model_output, name):
    x = GlobalAveragePooling2D()(base_model_output)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(4, name=name, activation='sigmoid')(x)
    return x
 
# Train the classification model for a specific attribute (type or manufacturer)
def train_classification(annotations, epochs, attribute, num_classes): 
    # Split data to get validation set ensuring proportional class distribution
    train_df, val_df = train_test_split(
        annotations, 
        test_size=0.2, 
        stratify=annotations[attribute],
        random_state=42
    )

    # Prepare datasets
    train_ds = prepare_dataset(train_df, attribute, isTraining=True)
    val_ds = prepare_dataset(val_df, attribute, isTraining=False)

    # Prepare the model
    base_model = prepare_base_model()
    model = Model(inputs=base_model.input, outputs=prepare_classification_head(base_model.output, num_classes, 'classification_output'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss= 'categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds
    )

    model.save(attribute + '_trained.keras') # Save the model

# Train the localization model for bounding box detection task
def train_localization(annotations, epochs): 
    # Split data to get validation set
    train_df, val_df = train_test_split(
        annotations, 
        test_size=0.20, 
        random_state=42
    )

    # Prepare datasets
    train_ds = prepare_dataset(train_df, isTraining=True)
    val_ds = prepare_dataset(val_df, isTraining=False)

    # Prepare the model
    base_model = prepare_base_model()
    model = Model(inputs=base_model.input, outputs=prepare_localization_head(base_model.output, 'detection_output'))
    
    # Fine-tune the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mean_squared_error',
        metrics=['mse']
    )

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds
    )

    model.save('localization_trained.keras') # Save the model

# Train the multi-task model for type classification, manufacturer classification, and localization tasks
def train_mtl(annotations, epochs):
    # Combine the classification attributes for evenly distributed classes on validation set
    annotations['combined'] = annotations[ATTRIBUTE_TYPE].astype(str) + '_' + annotations[ATTRIBUTE_MANUFACTURER].astype(str)
    annotations['combined'] = pd.factorize(annotations['combined'])[0]

    # Split data to get validation set ensuring proportional class distribution
    train_df, val_df = train_test_split(
        annotations,
        test_size=0.2,
        stratify=annotations['combined'],  
        random_state=42
    )

    # Prepare datasets
    train_ds = prepare_dataset(train_df, isTraining=True, isMultiTask=True)
    val_ds = prepare_dataset(val_df, isTraining=False, isMultiTask=True)

    # Prepare the base model
    base_model = prepare_base_model()

    x = base_model.output
    type_classification_head = prepare_classification_head(x, ATTRIBUTE_TYPE_COUNT, 'type_classification_output')
    manufacturer_classification_head = prepare_classification_head(x, ATTRIBUTE_MANUFACTURER_COUNT, 'manufacturer_classification_output')
    localization_head = prepare_localization_head(x, 'detection_output')

    model = Model(inputs=base_model.input, outputs=[type_classification_head, manufacturer_classification_head, localization_head])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            'type_classification_output': 'categorical_crossentropy',	
            'manufacturer_classification_output':  'categorical_crossentropy',
            'detection_output': 'mean_squared_error',
        },
        metrics={
            'type_classification_output': 'accuracy',
            'manufacturer_classification_output': 'accuracy',
            'detection_output': 'mse',
        },
        loss_weights={
            'type_classification_output': 1.0,
            'manufacturer_classification_output': 1.0,
            'detection_output': 10.0,
        }
    )

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=2
    )

    # Save the Model
    model.save('multitask_trained.keras')

def main():
    annotations = pd.read_csv(LABELS_FILENAME_NORMALIZED, sep=';') # Load the labels data 

    # Train type classification model
    train_classification(annotations, 35, ATTRIBUTE_TYPE, ATTRIBUTE_TYPE_COUNT)
    
    # Train manufacturer classification model
    train_classification(annotations, 45, ATTRIBUTE_MANUFACTURER, ATTRIBUTE_MANUFACTURER_COUNT)
    
    # Train localization model
    train_localization(annotations, 30)

    # Train multi-task model
    train_mtl(annotations, 45)

if __name__ == "__main__":
    main()
