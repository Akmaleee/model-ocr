import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Fungsi untuk membuat DataFrame dari direktori dataset
def directory_to_df(path: str) -> pd.DataFrame:
    data = []
    for class_name in os.listdir(path):
        class_folder = os.path.join(path, class_name)
        if os.path.isdir(class_folder):
            for file_name in os.listdir(class_folder):
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_folder, file_name)
                    data.append({'image': image_path, 'label': class_name})
    return pd.DataFrame(data)

# Buat DataFrame
df_path = './dataset'  # Ubah ke path dataset Anda
df = directory_to_df(df_path)
print(df.head())  # Pastikan kolom 'image' dan 'label' ada

# Data generators
train_gen = ImageDataGenerator(
    brightness_range=[0.7, 1.3],
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)
valid_gen = ImageDataGenerator(fill_mode='nearest')

# Data generator flows
train_gen_flow = train_gen.flow_from_dataframe(
    df,
    x_col='image',
    y_col='label',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'
)
valid_gen_flow = valid_gen.flow_from_dataframe(
    df,
    x_col='image',
    y_col='label',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Fungsi untuk membangun model tanpa batch_size pada Input
def build_crnn(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='Input')

    # Convolutional layers
    x = Conv2D(3, (3, 3), strides=1, activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(1024, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    # Fully connected layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Dense output
    outputs = Dense(num_classes, activation='softmax')(x)

    # Build the model
    model = Model(inputs, outputs)
    return model

# Hyperparameters
IMG_SIZE = (32, 32, 3)
BATCH_SIZE = 32
EPOCHS = 100

# Update NUM_CLASSES berdasarkan generator
NUM_CLASSES = len(train_gen_flow.class_indices)

# Build dan compile model
CNN_model = build_crnn(IMG_SIZE, NUM_CLASSES)
CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fungsi untuk membuat callbacks
def create_callbacks(model_name):
    early_stopping = EarlyStopping(patience=10, min_delta=0.01, verbose=1)
    reduce_lr = ReduceLROnPlateau(patience=5, min_delta=0.01, factor=0.5, verbose=1)
    model_checkpoint = ModelCheckpoint(f'{model_name}_model.keras', verbose=1, save_best_only=True)
    return [early_stopping, reduce_lr, model_checkpoint]

# Callback untuk debugging bentuk batch
class DebugCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        print(f"Starting batch {batch}")
    def on_train_batch_end(self, batch, logs=None):
        print(f"Finished batch {batch}")

# Model training
history = CNN_model.fit(
    train_gen_flow,
    epochs=EPOCHS,
    validation_data=valid_gen_flow,
    callbacks=create_callbacks("CustomCnn")
)

# Save the trained model
CNN_model.save("CustomCnn_model.h5")
print("Model telah disimpan sebagai CustomCnn_model.h5")
