import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
# from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator

labels = ['Healthy', 'PowderyMildew', 'BacterialBlight', 'NitrogenDeficiency',  'PotassiumDeficiency']

def get_data(path):
    x = []
    y = []
    for label in labels:
        images_path = os.path.normpath(os.path.join(path, label))
        if not os.path.exists(images_path):
            print(f"Path does not exist: {images_path}")
            continue
        for image in os.listdir(images_path):
            img_path = os.path.normpath(os.path.join(images_path, image))
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x.append(img)
            y.append(labels.index(label))

    x = np.array(x) / 255.0
    y = np.array(y)
    return x, y


train_path = './train'
test_path = './test'
val_path = './val'

train_x, train_y = get_data(train_path)
test_x, test_y = get_data(test_path)
val_x, val_y = get_data(val_path)

# Initialize ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(train_x)

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(datagen.flow(train_x, train_y, batch_size=32), epochs=10, validation_data=(val_x, val_y))

# Evaluate model
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test accuracy:', test_acc)

# Prediction
new_image = cv2.imread('./healthy42.jpg')
new_image = cv2.resize(new_image, (224, 224))
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
new_image = np.expand_dims(new_image, axis=0) / 255.0
prediction = model.predict(new_image)
predicted_class = labels[np.argmax(prediction)]
print('Predicted class:', predicted_class)

# Test accuracy: 0.32941177487373352
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 318ms/step
# Predicted class: BacterialBlight

