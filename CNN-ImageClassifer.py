#This program neeed Keras and Numpy!

#import libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Dense   

#--------------------------------------------
#Initialization
classifier = Sequential()

#--------------------------------------------
#Convolution (First Layer)
classifier.add(Conv2D(32,(3,3), input_shape = (64,64, 3), activation = 'relu'))

#input_shape is the image resolution, 64x64 is enough to make it above 90% accurate
#depending on how many training image you have and how powerful your hardware is.
#Activation is the type of activation you want, please read the documentation of CNN for
#more activation and details
#-------------------------------------------------

#Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#--------------------------------------------------
#Add second convolutional Layer
classifier.add(Conv2D(32,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#---------------------------------------------------
#Flattening
classifier.add(Flatten())

#---------------------------------------------------
#Full connection
classifier.add(Dense(units=128, activation = 'relu'))
classifier.add(Dense(units = 1 , activation = 'sigmoid'))

#---------------------------------------------------
#Compiling CNN
classifier.compile(optimizer = 'adams', loss='binary_crossentropy', metrics = ['accuracy'])

#----------------------------------------------------
#Fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

#-----------------------------------------------------
#Build Training Datagen
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

#-----------------------------------------------------
#Build Test Datagen
test_datagen = ImageDataGenerator(resclae= 1./255)

#-----------------------------------------------------
#Import Training Image
training_set = train_datagen.flow_from_directory('<Input your Training Directory here>',
target_size = (64,64),
batch_size = 32,
class_mode = 'binary')

#-----------------------------------------------------
#Import Test Image
test_set = test_datagen.flow_from_directory('<Input your Test Directory here',
target_size = (64,64),
batch_szie = 32,
class_mode = 'binary')

#-----------------------------------------------------
#Training and Test Start here
classifier.fit_generator(training_set,
steps_per_epoch = 12000,
epochs = 2,
validation_data = test_set,
validation_steps=200)

#Step per epoch actually depends on how many training images you have,
#more images mean more accuracy
#Epochs mean how many training you want
#Validation data mean the variabel where you input your test images
#Validation Steps mean how many validation you want 

#-----------------------------------------------------
#Prediction or Testing the Result
import numpy as np       
from keras.preprocessing import image

#input test Image
test_image = image.load_image('<Input your Test Image Directory Here',
target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1 :
    prediction = '<Name your result here>'
else :
    prediction = '<Name your result here>'

#----------------------------------------------------
#Congratulations, you have created the CNN!