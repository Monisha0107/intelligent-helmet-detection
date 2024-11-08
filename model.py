import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks

# Path to the dataset folder
dataset_path = 'processed_dataset'

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80-20 split
)

# Training data generator
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation data generator
val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Load pre-trained MobileNetV2, excluding the final classification layer
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base layers for initial training

# Add new layers for binary classification (helmet/no helmet)
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)  # Pooling layer to reduce dimensions
x = layers.Dense(1, activation='sigmoid')(x)  # Binary classification layer

# Build the model
model = models.Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback to prevent overfitting
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Initial training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[early_stopping]
)

# Fine-tune by unfreezing top layers and reducing learning rate
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Keep lower layers frozen
    layer.trainable = False

# Compile model with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model
fine_tune_history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stopping]
)

# Save the model
model.save('helmet-1.h5')

print("Model training complete and saved as 'helmet-detection-model.h5'")

val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")


from sklearn.metrics import accuracy_score
import numpy as np

# Predict on validation data
val_data.reset() 
predictions = model.predict(val_data)
predicted_classes = np.where(predictions > 0.5, 1, 0)  

# True labels
true_classes = val_data.classes

# Calculate accuracy score
accuracy = accuracy_score(true_classes, predicted_classes)
print(f"Calculated Accuracy Score: {accuracy * 100:.2f}%")