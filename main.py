import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, reshape, flatten
from tflearn.layers.estimator import regression
%matplotlib inline

MODEL_NAME = 'Soda_bottle_recognition'
IMG_SIZE = 128

#----------------
# import the data
#----------------
import pandas as pd
from sklearn.utils import shuffle
df = pd.read_csv("train.csv")
df = shuffle(df)

#--------------------------------------
# Convert the soda label text to number
#--------------------------------------

def convert_label_names(label):
    if label == 'M.Beer':
        return np.array([1,0,0,0,0,0,0,0])
    elif label == 'MD.Diet':
        return np.array([0,2,0,0,0,0,0,0])
    elif label == 'MD.Orig':
        return np.array([0,0,3,0,0,0,0,0])
    elif label == 'P.Cherry':
        return np.array([0,0,0,4,0,0,0,0])
    elif label == 'P.diet':
        return np.array([0,0,0,0,5,0,0,0])
    elif label == 'P.Orig':
        return np.array([0,0,0,0,0,6,0,0])
    elif label == 'P.Rsugar':
        return np.array([0,0,0,0,0,0,7,0])
    elif label == 'P.Zero':
        return np.array([0,0,0,0,0,0,0,8])
	
#--------------------------------------
# Read the image and resize accordingly
#--------------------------------------	
def create_data():
    data = []
    for i, row in df.iterrows():
        img_data = cv2.imread(row.Filename, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        data.append([np.array(img_data), convert_label_names(row.Label)])
    np.save('train_data.npy', data)
    return data

#-----------------------
#Create or load the data
#-----------------------
data = create_data()

#data = np.load('train_data.npy')

#-----------------------------------------------
# Divide the data into 80% training and 20% test
#-----------------------------------------------
train_data = data[:-1323]
test_data = data[-1323:]

X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#X_train = np.array([i[0] for i in train_data])
y_train = [i[1] for i in train_data]

X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#X_test = np.array([i[0] for i in test_data])
y_test = [i[1] for i in test_data]

#------------------------------------------
# Convolutional Neural Network architecture
# create the model
#------------------------------------------
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.5)
convnet = fully_connected(convnet, 8, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')

#------------------------
# Train and Fit the model
#------------------------
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, 
          validation_set=({'input': X_test}, {'targets': y_test}), 
          batch_size=52, show_metric=True, run_id=MODEL_NAME)
		  
#-----------------------
# Save the trained model
#-----------------------	  
model.save('my-model.tflearn')