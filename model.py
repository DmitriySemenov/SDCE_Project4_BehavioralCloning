import csv
import matplotlib.pyplot as plt 
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.models import load_model

from os import path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from math import ceil

samples = []    

with open('my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    print('Loading training data')
    for line in reader:
        if line[0] != 'center':
            samples.append(line)
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# The reason number is multiplied by 4 is that we use all three camera images 
# and the flipped center camera image
print('Number of training samples:',len(train_samples)*4)
print('Number of validation samples:',len(validation_samples)*4)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steer_angles = []
            for batch_sample in batch_samples:
                center_path = batch_sample[0]
                left_path = batch_sample[1]
                right_path = batch_sample[2]
            
                cent_filename = center_path.split('\\')[-1]
                center_path = 'my_data/IMG/' + cent_filename
                left_filename = left_path.split('\\')[-1]
                left_path = 'my_data/IMG/' + left_filename
                right_filename = right_path.split('\\')[-1]
                right_path = 'my_data/IMG/' + right_filename
                
                correction = 0.30
                
                image = plt.imread(center_path)
                image_flipped = np.fliplr(image)
                image_left = plt.imread(left_path)
                image_right = plt.imread(right_path)
                
                images.append(image)
                images.append(image_flipped)
                images.append(image_left)
                images.append(image_right)
                
                steering_center = float(batch_sample[3])
                steering_center_flipped = -steering_center
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                steer_angles.append(steering_center)
                steer_angles.append(steering_center_flipped)
                steer_angles.append(steering_left)
                steer_angles.append(steering_right)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(steer_angles)
            yield shuffle(X_train, y_train)

# Model Setup
if path.isfile('model.h5'):
    model = load_model('model.h5')
    print('Existing model loaded')
else:
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((60,20), (0,0))))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    print('New model created')

# Batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

history_object = model.fit_generator(train_generator, 
                    steps_per_epoch=ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator, 
                    validation_steps=ceil(len(validation_samples)/batch_size), 
                    epochs=3, verbose=2)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()