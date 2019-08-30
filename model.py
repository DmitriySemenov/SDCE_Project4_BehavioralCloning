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
    
images = []
steer_angles = []

with open('my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    print('Loading training data')
    for line in reader:
        if (line[0] != 'center'):
            
            center_path = line[0]
            left_path = line[1]
            right_path = line[2]
            
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
            
            steering_center = float(line[3])
            steering_center_flipped = -steering_center
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            
            steer_angles.append(steering_center)
            steer_angles.append(steering_center_flipped)
            steer_angles.append(steering_left)
            steer_angles.append(steering_right)
        
X_train = np.array(images)
y_train = np.array(steer_angles)
print('Training data loaded')

plt.hist(steer_angles, bins = 30)
plt.title('Steering angle distribution')
plt.show()

if path.isfile('model.h5'):
    model = load_model('model.h5')
    print('Existing model loaded')
else:
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((60,20), (0,0))))
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    print('New model created')

history_object = model.fit(X_train, y_train, validation_split=0.2, 
                           shuffle=True, nb_epoch = 5, verbose = 1)
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