from locale import normalize
import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

feature_list = pickle.load(open('embeddings.pkl','rb')) 
filenames = pickle.load(open('filenames.pkl','rb'))


model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D() 
])

img = image.load_img('sample/shoes.jpg',target_size = (224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis = 0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')

#input list
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

for file in indices[0]:
    print(filenames[file])