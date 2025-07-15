import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ====================== ARGUMENT PARSER ======================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--train_dir', type=str, default='/content/non-and-biodegradable-waste-dataset', help='Path to dataset')
args = parser.parse_args()

# ====================== DATA PREPARATION ======================
base_dir = r"C:\Users\AMAN REDDY\Downloads\Robotics\all files\Dataset.1"
train_dirs = [os.path.join(base_dir, f'TRAIN.{i}') for i in range(1, 5)]

def create_dataframe(train_dirs):
    image_paths = []
    labels = []
    for train_dir in train_dirs:
        for class_name in ['B', 'N']:
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
                        labels.append(class_name)
    return pd.DataFrame({'filename': image_paths, 'class': labels})

train_df = create_dataframe(train_dirs)

# ====================== DATA GENERATORS ======================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=args.batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=args.batch_size,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

# ====================== MODEL ARCHITECTURE ======================
def build_resnet():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_resnet()

# ====================== TRAINING WITH BENCHMARK ======================
start_time = time.time()
history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=args.epochs,
    verbose=1
)
end_time = time.time()
training_time = end_time - start_time

# ====================== CONFUSION MATRIX ======================
y_true = val_gen.classes
y_pred = np.argmax(model.predict(val_gen), axis=1)
class_labels = list(train_gen.class_indices.keys())
conf_matrix = confusion_matrix(y_true, y_pred)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix')
plt.savefig('confusion_matrix_normalized.png')

# ====================== SAVE RESULTS ======================
pd.DataFrame({
    'Epoch': list(range(1, args.epochs + 1)),
    'Train Accuracy': history.history['accuracy'],
    'Val Accuracy': history.history['val_accuracy'],
    'Train Loss': history.history['loss'],
    'Val Loss': history.history['val_loss']
}).to_csv('results.csv', index=False)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.legend()
plt.savefig('results.png')

# ====================== PRINT BENCHMARKS ======================
print("\nTraining Time: {:.2f} seconds".format(training_time))
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))

# ====================== SAVE MODEL ======================
model.save('waste_classifier_resnet50.h5')
print("\nModel saved successfully!")

if __name__ == "__main__":
    model.fit(train_gen)
