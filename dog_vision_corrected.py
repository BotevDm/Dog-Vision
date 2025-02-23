# End-to-end Multi-class Dog Breed Classification

## 1. Problem
# Identifying the breed of a dog given an image.

## 2. Data
# https://www.kaggle.com/c/dog-breed-identification/data

## 3. Evaluation
# Evaluation is a file with probabilities for each dog breed of each test image.

## 4. Features
# * Unstructured data
# * 120 dog breeds (120 classes)
# * 10,000+ images in training and test sets, no labels for test set

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from IPython.display import Image
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score

print('TF version:', tf.__version__)
print('TF Hub version:', hub.__version__)

# Check for GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
print("GPU", 'available' if gpu_devices else 'not available')
if not gpu_devices:
    print("Warning: No GPU found, running on CPU.")

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 100
MODEL_URL = "https://kaggle.com/models/google/mobilenet-v2/TensorFlow2/130-224-classification/1"

# Load labels
try:
    labels_csv = pd.read_csv('drive/MyDrive/Dog Vision/labels.csv')
    print(labels_csv.describe())
    labels_csv.head()
except FileNotFoundError:
    raise FileNotFoundError("Could not find labels.csv at the specified path.")

# Visualize breed distribution
labels_csv['breed'].value_counts().plot.bar(figsize=(20, 10))
plt.title("Distribution of Dog Breeds")
plt.show()

# Prepare filenames and labels
filenames = ['drive/MyDrive/Dog Vision/train/' + fname + '.jpg' for fname in labels_csv['id']]
labels = labels_csv['breed'].to_numpy()
unique_breeds = np.unique(labels)
boolean_labels = [label == unique_breeds for label in labels]

# Validate data integrity
if len(filenames) != len(os.listdir('drive/MyDrive/Dog Vision/train')):
    print("Warning: Number of filenames does not match number of files in directory.")
if len(labels) != len(filenames):
    raise ValueError("Number of labels does not match number of filenames.")

# Display sample image
Image(filenames[9000])

# Split data into training and validation sets
NUM_IMAGES = 1000  # Adjustable via experimentation
X_train, X_val, y_train, y_val = train_test_split(filenames[:NUM_IMAGES],
                                                  boolean_labels[:NUM_IMAGES],
                                                  test_size=0.2,
                                                  random_state=42)

# Image preprocessing function
def process_image(image_path, img_size=IMG_SIZE):
    """Converts an image file into a preprocessed Tensor."""
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [img_size, img_size])
        return image
    except tf.errors.InvalidArgumentError:
        print(f"Warning: Could not process image at {image_path}")
        return None

# Create image-label tuple
def get_image_label(image_path, label):
    image = process_image(image_path)
    if image is None:
        return None, None
    return image, label

# Batch data creation
def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """Creates batches of image-label pairs."""
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = data.map(process_image).batch(batch_size)
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = data.map(get_image_label).filter(lambda x, y: x is not None).batch(batch_size)
    else:
        print("Creating training data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.map(get_image_label).filter(lambda x, y: x is not None).batch(batch_size)
    return data_batch

# Create batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)

# Visualize batches
def show_images(images, labels, max_images=25):
    """Displays up to max_images from a batch with their labels."""
    num_images = min(len(images), max_images)
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(unique_breeds[labels[i].argmax()])
        plt.axis('off')
    plt.show()

train_images, train_labels = next(train_data.as_numpy_iterator())
show_images(train_images, train_labels)

val_images, val_labels = next(val_data.as_numpy_iterator())
show_images(val_images, val_labels)

# Custom metric for recall (macro-average)
def macro_recall(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    return tf.py_function(lambda yt, yp: recall_score(yt, yp, average='macro'),
                          [y_true, y_pred], tf.float32)

# Model creation with additional metrics
def create_model(input_shape=(None, IMG_SIZE, IMG_SIZE, 3), output_shape=len(unique_breeds), model_url=MODEL_URL):
    """Creates and compiles a transfer learning model with custom metrics."""
    print(f'Building model with: {model_url}')
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url, trainable=False, input_shape=input_shape[1:]),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy', macro_recall, tf.keras.metrics.AUC(multi_label=True, name='roc_auc')])
    return model

# Callbacks
def create_tensorboard_callback():
    logdir = os.path.join('drive/MyDrive/Dog Vision/logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    return tf.keras.callbacks.TensorBoard(logdir)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

# Training function
def train_model():
    """Trains and returns a model."""
    model = create_model()
    tensorboard = create_tensorboard_callback()
    model.fit(x=train_data,
              epochs=NUM_EPOCHS,
              validation_data=val_data,
              validation_freq=1,
              callbacks=[tensorboard, early_stopping])
    return model

model = train_model()

# Evaluate metrics on validation set
def evaluate_model(model, val_data):
    """Evaluates the model on validation data and prints detailed metrics."""
    # Collect true labels and predictions
    y_true = np.concatenate([y for _, y in val_data], axis=0)
    y_pred_probs = model.predict(val_data, verbose=1)
    
    # Convert to class indices
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # Compute metrics
    accuracy = np.mean(y_true_classes == y_pred_classes)
    recall_macro = recall_score(y_true_classes, y_pred_classes, average='macro')
    roc_auc_macro = roc_auc_score(y_true, y_pred_probs, average='macro', multi_class='ovr')
    
    # Print results
    print("\nValidation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-average Recall: {recall_macro:.4f}")
    print(f"Macro-average ROC-AUC: {roc_auc_macro:.4f}")
    
    return {'accuracy': accuracy, 'recall': recall_macro, 'roc_auc': roc_auc_macro}

metrics = evaluate_model(model, val_data)

# Predictions with label conversion
def get_pred_label(prediction_probabilities):
    """Converts prediction probabilities to a label."""
    return unique_breeds[np.argmax(prediction_probabilities)]

# Example prediction
index = 42
pred_label = get_pred_label(model.predict(val_data)[index])
print(f'\nExample Prediction (index {index}):')
print(f'Max probability: {np.max(model.predict(val_data)[index]):.4f}')
print(f'Predicted breed: {pred_label}')