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

image_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.5
)

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(img_height, img_width), class_mode='binary')
augment_image = [train_data_gen[0][0][1] for i in range(5)]
plotImage(augment_image)

image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size, directory=validation_dir, target_size=(img_height, img_width), class_mode='binary')

model_new = Sequential([
    Conv2D(filters=16, kernel_size=3, padding='same', strides=1, activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=2),
    Dropout(rate=0.2),
    Conv2D(filters=32, kernel_size=3, padding='same', strides=1, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=1)
])

model_new.compile(optimizer = 'adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

print(model_new.summary())

history = model_new.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_train // batch_size
)

model_new.save('./Model/model_with_image_gen.keras')

# 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Train acc')
# plt.plot(epochs_range, val_acc, label='Train val_acc')
plt.legend(loc='best')
plt.title('Train, Validation acc')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Train loss')
# plt.plot(epochs_range, val_loss, label='Train val_loss')
plt.legend(loc='best')
plt.title('Train, Validation loss')

plt.show()