from typing import Any
from PIL import Image, ImageStat
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import joblib
from keras.models import load_model


#Processing the trainset
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('chest_xray/train',
                                                 target_size = (200, 200),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

#Processing the testset
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('chest_xray/train',
                                            target_size = (200, 200),
                                            batch_size = 32,
                                            class_mode = 'categorical')
from matplotlib.pyplot import imshow



#Initialising the CNN model
cnn = tf.keras.models.Sequential()
#Convolutional feature mapping and rectifier activation --> correcting linearity
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[200, 200, 3]))
#Pooling --> reducing features to avoid overfitting
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
#Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
#Flattening
cnn.add(tf.keras.layers.Flatten())
#Full Connection
cnn.add(tf.keras.layers.Dense(units=100, activation='relu'))
#Output Layer
cnn.add(tf.keras.layers.Dense(units=2, activation='softmax'))
#cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#Compiling the CNN model
cnn.compile(optimizer = 'adam',loss= 'binary_crossentropy',  metrics = ['accuracy'])
#Training the CNN model
cnn.fit(x = training_set, validation_data = test_set, epochs = 6)
# save the model to disk
cnn.save("cnn.h5")

print ("=============================================================================================")

import numpy as np
from keras.preprocessing import image

nbp_normal = 0
nbn_normal = 0
nbp_pneumonie = 0
nbn_pneumonie = 0

#examples = ['chest_xray/test/NORMAL/NORMAL-745902-0001.jpeg',"chest_xray/test/NORMAL/NORMAL-11419-0001.jpeg","chest_xray/test/NORMAL/NORMAL-9896495-0001.jpeg", 'chest_xray/test/PNEUMONIA/BACTERIA-227418-0001.jpeg', 'chest_xray/train/PNEUMONIA/BACTERIA-7422-0001.jpeg']
loaded_model = load_model("cnn.h5")
for filename in glob.glob('chest_xray/test/NORMAL/*.jpeg'):
    ex = tf.keras.utils.load_img(filename, target_size=(200, 200))
    ex = np.expand_dims(tf.keras.preprocessing.image.img_to_array(ex), axis=0)

    prediction = loaded_model.predict(ex)
    
    result = loaded_model.predict(ex)

    if np.argmax(result) == 0:
        prediction_ = 'Non Pneumonia'
        nbp_normal = nbp_normal + 1
    else:
        prediction_ = 'Pneumonia'
        nbn_normal = nbn_normal + 1
    """
    if result[0][0] == 1:
        prediction = 'Pneumonia'
        
    else:
        prediction = 'Non Pneumonia'
    """

    print(prediction_)

print ("=============================================================================================")
print ("=============================================================================================")
print ("=============================================================================================")

for filename in glob.glob('chest_xray/test/PNEUMONIA/*.jpeg'):
    ex = tf.keras.utils.load_img(filename, target_size=(200, 200))
    ex = np.expand_dims(tf.keras.preprocessing.image.img_to_array(ex), axis=0)

    prediction = loaded_model.predict(ex)
    
    result = loaded_model.predict(ex)

    if np.argmax(result) == 0:
        prediction_ = 'Non Pneumonia'
        nbp_normal = nbp_pneumonie + 1
    else:
        prediction_ = 'Pneumonia'
        nbn_normal = nbn_pneumonie + 1
    """
    if result[0][0] == 1:
        prediction = 'Pneumonia'
        nbp_pneumonie =  nbp_pneumonie + 1
    else:
        prediction = 'Non Pneumonia'
        nbn_pneumonie =  nbn_pneumonie + 1
    """
    print(prediction_)

print ("=============================================================================================")

print ("nombre de pneumonie detectés dans le dossier normal : ", nbp_normal)
print ("nombre de normal detectés dans le dossier normal : ", nbn_normal)
print ("nombre de pneumonie detectés dans le dossier pneumonie : ", nbp_pneumonie)
print ("nombre de normal detectés dans le dossier pneumonie : ", nbn_pneumonie)
