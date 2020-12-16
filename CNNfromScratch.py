from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Dense
import matplotlib.pyplot as plt
from glob import glob
from keras.layers.advanced_activations import ReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D


train_path = "eyes_dataset/train/"
valid_path = "eyes_dataset/test"

# Show an example image from dataset
img = load_img(train_path + "Open/_5.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)

print(x.shape)

# How many classes(Open-Closed)
numberOfClass = len(glob(train_path+"/*"))
print(numberOfClass)

# Data Augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(24, 24),
        batch_size=32,
        class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
        valid_path,
        target_size=(24, 24),
        batch_size=32,
        class_mode='categorical')


# Building CNN Model

model = Sequential()


model.add(Convolution2D(32, (3,3), activation = 'relu', padding='same', use_bias=False, input_shape=(24,24,3)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3,3), activation = 'relu', padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), activation = 'relu', padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3), activation = 'relu', padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2,activation = 'softmax'))
model.summary()

# Compile the model

model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['acc'])

# Fit the model

batch_size = 32
# 7000 =>number of images in train set, 700 => number of images in test set
model.fit(
        train_generator,
        steps_per_epoch=7000/32,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=700/32)

# Save the model to use later
model.save("DrawsinessDetectionModel.h5")

