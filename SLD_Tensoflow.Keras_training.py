import time  #hacktoberfest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
from glob import glob

#import tensorflow as tf
#from tensorflow.keras.backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 1
#set_session(tf.Session(config=config))

# Imports for Deep Learning
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
#from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = "dataset/training_set"
target_size = (64, 64)
target_dims = (64, 64, 3) # add channel for RGB
n_classes = 29
val_frac = 0.1
batch_size = 64

CLASSES = [folder[len(data_dir) + 1:] for folder in glob(data_dir + '/*')]
CLASSES.sort()

def plot_one_sample_of_each(base_path):
    cols = 5
    rows = int(np.ceil(len(CLASSES) / cols))
    fig = plt.figure(figsize=(16, 20))
    
    for i in range(len(CLASSES)):
        cls = CLASSES[i]
        img_path = base_path + '/' + cls + '/**'
        path_contents = glob(img_path)
    
        imgs = random.sample(path_contents, 1)

        sp = plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.imread(imgs[0]))
        plt.title(cls)
        sp.axis('off')

    plt.show()
    return
#plot_one_sample_of_each(data_dir)

data_augmentor = ImageDataGenerator(samplewise_center = True,
                                    samplewise_std_normalization = True,
                                    validation_split = val_frac)

train_generator = data_augmentor.flow_from_directory(data_dir,
                                                     target_size = target_size,
                                                     batch_size = batch_size,
                                                     shuffle = True,
                                                     subset = "training")

val_generator = data_augmentor.flow_from_directory(data_dir,
                                                   target_size = target_size,
                                                   batch_size = batch_size,
                                                   subset = "validation")

model = Sequential()
model.add(Conv2D(64, kernel_size = 4, strides = 1, activation = 'relu', input_shape = target_dims))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size = 4, strides = 2, activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size = 4, strides = 1, activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size = 4, strides = 2, activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(256, kernel_size = 4, strides = 1, activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(256, kernel_size = 4, strides = 2, activation = 'relu'))
#model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(n_classes, activation = 'softmax'))

model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ["accuracy"])

#model.fit(train_generator, epochs = 5, batch_size = 128)

start = time.perf_counter()
model.fit_generator(train_generator,
                   steps_per_epoch = 87000,
                   epochs = 5,
                   validation_data = val_generator,
                   validation_steps = 29)
end = time.perf_counter()
print("Execution Time: ", end - start)

#model.save("hash_file_3.h5")
#
#
#from keras.models import load_model
#model = load_model('hash_file_4.h5')
