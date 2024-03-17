import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

batch_size = 128
epochs = 30
img_height = 150
img_width = 150

path = './Dataset'
train_dir = os.path.join(path, 'train')
validation_dir = os.path.join(path, 'validation')
train_ddensukki = os.path.join(train_dir, 'ddensukki')
validation_ddensukki = os.path.join(validation_dir, 'ddensukki')
train_stone = os.path.join(train_dir, 'stone')
validation_stone = os.path.join(validation_dir, 'stone')

num_ddensukki_tr = len(os.listdir(train_ddensukki))
num_stone_tr = len(os.listdir(train_stone))
num_ddensukki_val = len(os.listdir(validation_ddensukki))
num_stone_val = len(os.listdir(validation_stone))

total_train = num_ddensukki_tr + num_stone_tr
total_val = num_stone_val + num_ddensukki_val

train_img_gen = ImageDataGenerator(rescale=1./255)
validation_img_gen = ImageDataGenerator(rescale=1./255)

train_data_gen = train_img_gen.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(img_height, img_width), class_mode='binary')
val_data_gen = validation_img_gen.flow_from_directory(batch_size=batch_size, directory=validation_dir, target_size=(img_height, img_width), class_mode='binary')

sample_train_images, _ = next(iter(train_data_gen))

def plotImage(img_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax in zip(img_arr, axes):
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

plotImage(sample_train_images)

model = Sequential([
    Conv2D(filters=16, kernel_size=3, padding='same', strides=1, activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=32, kernel_size=3, padding='same', strides=1, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(model.summary())

history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
model.save('./Model/model_without_image_gen.keras')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='best')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='best')
plt.title('Training and Validation Loss')
plt.show()

