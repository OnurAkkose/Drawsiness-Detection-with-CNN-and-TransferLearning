# We can use pretrained models trained on ImageNet. It is called Transfer Learning.
  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from glob import glob


train_path = "eyes_dataset/train/"
valid_path = "eyes_dataset/test"



numberOfClass = len(glob(train_path+"/*"))

# I chose VGG16 but there are many different pretrained models.(e.g. Resnet, Xception ...)
vgg = VGG16()

# VGG16s layers
print(vgg.summary())


vgg_layer_list = vgg.layers

model = Sequential()

# added VGG16 layers to our model sequential

for i in range(len(vgg_layer_list)-1):
    model.add(vgg_layer_list[i])


# we should freeze VGG16s layers
for layers in model.layers:
    layers.trainable = False

# added softmax to classify dataset on end of models layers
model.add(Dense(numberOfClass, activation="softmax"))

# compile the model
model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

# train the model. VGG16's input shape is 224x224
train_data = ImageDataGenerator().flow_from_directory(train_path,target_size = (224,224))
test_data = ImageDataGenerator().flow_from_directory(valid_path,target_size = (224,224))

batch_size = 32
# 7000 =>number of images in train set, 700 => number of images in test set
model.fit(train_data,
           steps_per_epoch=7000//batch_size,
           epochs= 25,
           validation_data=test_data,
           validation_steps= 700//batch_size)

# save the model
model.save("VGG16forDrawsinessDetection.h5")