import os
import cv2
import numpy
import imageio
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from keras import backend as K
import sys

K.set_image_dim_ordering('th')

image_rows, image_columns, image_depth = 64, 64, 18

training_list = []

negativepath = '../../Dataset/Negative/'
positivepath = '../../Dataset/Positive/'
surprisepath = '../../Dataset/Surprise/'

directorylisting = os.listdir(negativepath)
for video in directorylisting:
    videopath = negativepath + video
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)

directorylisting = os.listdir(positivepath)
for video in directorylisting:
    videopath = positivepath + video
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)

directorylisting = os.listdir(surprisepath)
for video in directorylisting:
    videopath = surprisepath + video
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)

training_list = numpy.asarray(training_list)
trainingsamples = len(training_list)
traininglabels = numpy.zeros((trainingsamples, ), dtype = int)

traininglabels[0:66] = 0
traininglabels[66:113] = 1
traininglabels[113:156] = 2

traininglabels = np_utils.to_categorical(traininglabels, 3)

training_data = [training_list, traininglabels]
(trainingframes, traininglabels) = (training_data[0], training_data[1])
training_set = numpy.zeros((trainingsamples, 1, image_rows, image_columns, image_depth))

for h in range(trainingsamples):
    training_set[h][0][:][:][:] = trainingframes[h, :, :, :]

training_set = training_set.astype('float32')
training_set -= numpy.mean(training_set)
training_set /= numpy.max(training_set)

# Save training images and labels in a numpy array
numpy.save('numpy_training_datasets/microexpstcnn_images.npy', training_set)
numpy.save('numpy_training_datasets/microexpstcnn_labels.npy', traininglabels)

# Load training images and labels that are stored in numpy array
"""
training_set = numpy.load('numpy_training_datasets/microexpstcnn_images.npy')
traininglabels =numpy.load('numpy_training_datasets/microexpstcnn_labels.npy')
"""

# MicroExpSTCNN Model
model = Sequential()
model.add(Convolution3D(32, (3, 3, 15), input_shape=(1, image_rows, image_columns, image_depth), activation='relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, init='normal'))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])

model.summary()

filepath="weights_microexpstcnn/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Load pre-trained weights
"""
model.load_weights('weights_microexpstcnn/weights-improvement-40-0.69.hdf5')
"""

# Spliting the dataset into training and validation sets
train_images, validation_images, train_labels, validation_labels =  train_test_split(training_set, traininglabels, test_size=0.2, random_state=4)

# Save validation set in a numpy array
numpy.save('numpy_validation_dataset/microexpstcnn_val_images.npy', validation_images)
numpy.save('numpy_validation_dataset/microexpstcnn_val_labels.npy', validation_labels)

# Load validation set from numpy array
"""
validation_images = numpy.load('numpy_validation_datasets/microexpstcnn_val_images.npy')
validation_labels = numpy.load('numpy_validation_dataset/microexpstcnn_val_labels.npy')
"""

# Training the model
hist = model.fit(train_images, train_labels, validation_data = (validation_images, validation_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 100, shuffle=True)

# Finding Confusion Matrix using pretrained weights
"""
predictions = model.predict(validation_images)
predictions_labels = numpy.argmax(predictions, axis=1)
validation_labels = numpy.argmax(validation_labels, axis=1)
cfm = confusion_matrix(validation_labels, predictions_labels)
print (cfm)
"""