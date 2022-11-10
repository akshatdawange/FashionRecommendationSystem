from copyreg import pickle
from unicodedata import normalize
from unittest import result
from xml.etree.ElementInclude import include
import tensorflow as tf

from tensorflow.keras.preprocessing import image #importing images
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

import numpy as np
from numpy.linalg import norm
import os
import pickle

#Creating the model
#Weights -> Obtained from training on imagenet dataset
#include_top = FALSE -> We are defining our own top layer
#input_shape -> Scaled down size of the images we feed to ResNet
model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

#Trainable -> False since we wouldn't be training the model as it is already being trained upon the 'imagenet' dataset
model.trainable = False

#Replacing the top layer in ResNet with GlobalMaxPooling2D
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D() 
])

#print(model.summary())

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size = (224,224)) #adds the image to your environment
    img_array = image.img_to_array(img) #converts image into an array
    expanded_img_array = np.expand_dims(img_array, axis = 0) #represents a single image as a batch of images since Keras can't process a single image 
    preprocessed_img = preprocess_input(expanded_img_array) #converts image from RGB to BGR and each color channel is zero centred wrt imagenet dataset without scaling
    result = model.predict(preprocessed_img).flatten() #displays the size in 1-Dimensional array
    normalized_result = result / norm(result)

    return normalized_result

#adding file paths with the file names 
filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images' , file)) #appends each path in the 'images' folder with the file name

print(len(filenames))
print(filenames[0:5])

#prints all the files in the directory
#print(os.listdir('images'))

#2D array to show features of each image (feature extraction)
feature_list = []
for file in filenames:
    feature_list.append(extract_features(file, model)) #will run the loop for 44k times and the return of the function 'extract features' will be appended to the feature list

pickle.dump(feature_list,open('embeddings.pkl', 'wb'))
pickle.dump(filenames,open('filenames.pkl', 'wb'))
