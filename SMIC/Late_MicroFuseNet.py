import os
import cv2
from keras import regularizers
import dlib
import numpy
import imageio
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.utils import multi_gpu_model
from keras.optimizers import SGD, RMSprop
from keras.layers import Concatenate, Input, concatenate, add, multiply, maximum
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import backend as K
K.set_image_dim_ordering('th')

# DLib Face Detection
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
	pass

class NoFaces(Exception):
	pass

def get_landmark(img):
	rects = detector(img, 1)
	if len(rects) > 1:
		pass
	if len(rects) == 0:
		pass
	ans = numpy.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
	return ans

def annotate_landmarks(img, landmarks, font_scale = 0.4):
	for idx, point in enumerate(landmarks):
		pos = (point[0, 0], point[0, 1])
		cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=font_scale, color=(0, 0, 255))
		cv2.circle(img, pos, 3, color=(0, 255, 255))
	return img

negativepath = '../../Dataset/HS/Negative/'
positivepath = '../../Dataset/HS/Positive/'
surprisepath = '../../Dataset/HS/Surprise/'

eye_training_list = []
nose_training_list = []

directorylisting = os.listdir(negativepath)
for video in directorylisting:
    videopath = negativepath + video
    eye_frames = []
    nose_mouth_frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           landmarks = get_landmark(image)
           numpylandmarks = numpy.asarray(landmarks)
           eye_image = image[numpylandmarks[19][1]:numpylandmarks[1][1], numpylandmarks[1][0]:numpylandmarks[15][0]]
           eye_image = cv2.resize(eye_image, (32, 32), interpolation = cv2.INTER_AREA)
           eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
           nose_mouth_image = image[numpylandmarks[2][1]:numpylandmarks[6][1], numpylandmarks[2][0]:numpylandmarks[14][0]]
           nose_mouth_image = cv2.resize(nose_mouth_image, (32, 32), interpolation = cv2.INTER_AREA)
           nose_mouth_image = cv2.cvtColor(nose_mouth_image, cv2.COLOR_BGR2GRAY)
           eye_frames.append(eye_image)
           nose_mouth_frames.append(nose_mouth_image)
    eye_frames = numpy.asarray(eye_frames)
    nose_mouth_frames = numpy.asarray(nose_mouth_frames)
    eye_videoarray = numpy.rollaxis(numpy.rollaxis(eye_frames, 2, 0), 2, 0)
    nose_mouth_videoarray = numpy.rollaxis(numpy.rollaxis(nose_mouth_frames, 2, 0), 2, 0)
    eye_training_list.append(eye_videoarray)
    nose_training_list.append(nose_mouth_videoarray)

directorylisting = os.listdir(positivepath)
for video in directorylisting:
    videopath = positivepath + video
    eye_frames = []
    nose_mouth_frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           landmarks = get_landmark(image)
           numpylandmarks = numpy.asarray(landmarks)
           eye_image = image[numpylandmarks[19][1]:numpylandmarks[1][1], numpylandmarks[1][0]:numpylandmarks[15][0]]
           eye_image = cv2.resize(eye_image, (32, 32), interpolation = cv2.INTER_AREA)
           eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
           nose_mouth_image = image[numpylandmarks[2][1]:numpylandmarks[6][1], numpylandmarks[2][0]:numpylandmarks[14][0]]
           nose_mouth_image = cv2.resize(nose_mouth_image, (32, 32), interpolation = cv2.INTER_AREA)
           nose_mouth_image = cv2.cvtColor(nose_mouth_image, cv2.COLOR_BGR2GRAY)
           eye_frames.append(eye_image)
           nose_mouth_frames.append(nose_mouth_image)
    eye_frames = numpy.asarray(eye_frames)
    nose_mouth_frames = numpy.asarray(nose_mouth_frames)
    eye_videoarray = numpy.rollaxis(numpy.rollaxis(eye_frames, 2, 0), 2, 0)
    nose_mouth_videoarray = numpy.rollaxis(numpy.rollaxis(nose_mouth_frames, 2, 0), 2, 0)
    eye_training_list.append(eye_videoarray)
    nose_training_list.append(nose_mouth_videoarray)

directorylisting = os.listdir(surprisepath)
for video in directorylisting:
    videopath = surprisepath + video
    eye_frames = []
    nose_mouth_frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           landmarks = get_landmark(image)
           numpylandmarks = numpy.asarray(landmarks)
           eye_image = image[numpylandmarks[19][1]:numpylandmarks[1][1], numpylandmarks[1][0]:numpylandmarks[15][0]]
           eye_image = cv2.resize(eye_image, (32, 32), interpolation = cv2.INTER_AREA)
           eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
           nose_mouth_image = image[numpylandmarks[2][1]:numpylandmarks[6][1], numpylandmarks[2][0]:numpylandmarks[14][0]]
           nose_mouth_image = cv2.resize(nose_mouth_image, (32, 32), interpolation = cv2.INTER_AREA)
           nose_mouth_image = cv2.cvtColor(nose_mouth_image, cv2.COLOR_BGR2GRAY)
           eye_frames.append(eye_image)
           nose_mouth_frames.append(nose_mouth_image)
    eye_frames = numpy.asarray(eye_frames)
    nose_mouth_frames = numpy.asarray(nose_mouth_frames)
    eye_videoarray = numpy.rollaxis(numpy.rollaxis(eye_frames, 2, 0), 2, 0)
    nose_mouth_videoarray = numpy.rollaxis(numpy.rollaxis(nose_mouth_frames, 2, 0), 2, 0)
    eye_training_list.append(eye_videoarray)
    nose_training_list.append(nose_mouth_videoarray)
    eye_training_list.append(eye_videoarray)
    nose_training_list.append(nose_mouth_videoarray)

eye_training_list = numpy.asarray(eye_training_list)
nose_training_list = numpy.asarray(nose_training_list)

eye_trainingsamples = len(eye_training_list)
nose_trainingsamples = len(nose_training_list)

eye_traininglabels = numpy.zeros((eye_trainingsamples, ), dtype = int)
nose_traininglabels = numpy.zeros((nose_trainingsamples, ), dtype = int)

eye_traininglabels[0:66] = 0
eye_traininglabels[66:113] = 1
eye_traininglabels[113:156] = 2

nose_traininglabels[0:66] = 0
nose_traininglabels[66:113] = 1
nose_traininglabels[113:156] = 2

eye_traininglabels = np_utils.to_categorical(eye_traininglabels, 3)
nose_traininglabels = np_utils.to_categorical(nose_traininglabels, 3)

etraining_data = [eye_training_list, eye_traininglabels]
(etrainingframes, etraininglabels) = (etraining_data[0], etraining_data[1])
etraining_set = numpy.zeros((eye_trainingsamples, 1, 32, 32, 18))
for h in range(eye_trainingsamples):
	etraining_set[h][0][:][:][:] = etrainingframes[h,:,:,:]

etraining_set = etraining_set.astype('float32')
etraining_set -= numpy.mean(etraining_set)
etraining_set /= numpy.max(etraining_set)

ntraining_data = [nose_training_list, nose_traininglabels]
(ntrainingframes, ntraininglabels) = (ntraining_data[0], ntraining_data[1])
ntraining_set = numpy.zeros((nose_trainingsamples, 1, 32, 32, 18))
for h in range(nose_trainingsamples):
        ntraining_set[h][0][:][:][:] = ntrainingframes[h,:,:,:]

ntraining_set = ntraining_set.astype('float32')
ntraining_set -= numpy.mean(ntraining_set)
ntraining_set /= numpy.max(ntraining_set)

numpy.save('numpy_training_datasets/late_microexpfusenetnoseimages.npy', ntraining_set)
numpy.save('numpy_training_datasets/late_microexpfuseneteyeimages.npy', etraining_set)
numpy.save('numpy_training_datasets/late_microexpfusenetnoselabels.npy', nose_traininglabels)
numpy.save('numpy_training_datasets/late_microexpfuseneteyelabels.npy', eye_traininglabels)

# Load training images and labels that are stored in numpy array
"""
ntraining_set = numpy.load('numpy_training_datasets/late_microexpfusenetnoseimages.npy')
etraining_set = numpy.load('numpy_training_datasets/late_microexpfuseneteyeimages.npy')
eye_traininglabels = numpy.load('numpy_training_datasets/late_microexpfusenetnoselabels.npy')
nose_traininglabels = numpy.load('numpy_training_datasets/late_microexpfuseneteyelabels.npy')
"""

# Late MicroExpFuseNet Model
eye_input = Input(shape = (1, 32, 32, 18))
eye_conv = Convolution3D(32, (3, 3, 15))(eye_input)
ract_1 = Activation('relu')(eye_conv)
maxpool_1 = MaxPooling3D(pool_size=(3, 3, 3))(ract_1)
ract_2 = Activation('relu')(maxpool_1)
dropout_1 = Dropout(0.5)(ract_2)
flatten_1 = Flatten()(dropout_1)
dense_1 = Dense(1024, )(flatten_1)
dropout_2 = Dropout(0.5)(dense_1)
dense_2= Dense(128, )(dropout_2)
dropout_3 = Dropout(0.5)(dense_2)

nose_input = Input(shape = (1, 32, 32, 18))
nose_conv = Convolution3D(32, (3, 3, 15))(nose_input)
ract_3 = Activation('relu')(nose_conv)
maxpool_2 = MaxPooling3D(pool_size=(3, 3, 3))(ract_3)
ract_4 = Activation('relu')(maxpool_2)
dropout_2 = Dropout(0.5)(ract_4)
flatten_2 = Flatten()(dropout_2)
dense_3 = Dense(1024, )(flatten_2)
dropout_4 = Dropout(0.5)(dense_3)
dense_4 = Dense(128, )(dropout_4)
dropout_5 = Dropout(0.5)(dense_4)

concat = Concatenate(axis = 1)([dropout_3, dropout_5])

dense_5 = Dense(3, )(concat)
activation = Activation('softmax')(dense_5)

model = Model(inputs = [eye_input, nose_input], outputs = activation)
model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])

filepath="weights_late_microexpfusenet/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.summary()

# Load pre-trained weights
"""
model.load_weights('weights_late_microexpfusenet/weights-improvement-22-0.83.hdf5')
"""

# Spliting the dataset into training and validation sets
etrain_images, evalidation_images, etrain_labels, evalidation_labels =  train_test_split(etraining_set, eye_traininglabels, test_size=0.2, shuffle=False)
ntrain_images, nvalidation_images, ntrain_labels, nvalidation_labels =  train_test_split(ntraining_set, nose_traininglabels, test_size=0.2, shuffle=False)

# Save validation set in a numpy array
numpy.save('numpy_validation_datasets/late_microexpfusenet_eval_images.npy', evalidation_images)
numpy.save('numpy_validation_datasets/late_microexpfusenet_nval_images.npy', nvalidation_images)
numpy.save('numpy_validation_datasets/late_microexpfusenet_eval_labels.npy', evalidation_labels)
numpy.save('numpy_validation_datasets/late_microexpfusenet_nval_labels.npy', nvalidation_labels)

# Training the model
history = model.fit([etrain_images, ntrain_images], etrain_labels, validation_data = ([etraining_set, ntraining_set], eye_traininglabels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 250, shuffle=True)

# Loading Load validation set from numpy array
"""
eimg = numpy.load('numpy_validation_datasets/late_microexpfusenet_eval_images.npy')
nimg = numpy.load('numpy_validation_datasets/late_microexpfusenet_nval_images.npy')
labels = numpy.load('numpy_validation_datasets/late_microexpfusenet_eval_labels.npy')
"""

# Finding Confusion Matrix using pretrained weights
"""
predictions = model.predict([eimg, nimg])
predictions_labels = numpy.argmax(predictions, axis=1)
validation_labels = numpy.argmax(labels, axis=1)
cfm = confusion_matrix(validation_labels, predictions_labels)
print (cfm)
"""
