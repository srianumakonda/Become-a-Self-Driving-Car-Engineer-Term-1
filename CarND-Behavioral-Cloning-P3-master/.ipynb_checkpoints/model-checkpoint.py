import cv2
import csv
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lines = []

with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines[1:]:
    source_path = str("C:/Users/srian/Desktop/Become-a-Self-Driving-Car-Engineer/CarND-Behavioral-Cloning-P3-master/"+line[0])
    image = cv2.imread(source_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imwrite("1.jpg",image)
    images.append(image)
    images.append(cv2.flip(image,1))
    cv2.imwrite("2.jpg",cv2.flip(image,1))
    break
    measurements.append(float(line[3])-float(0.25))
    measurements.append((float(line[3])-float(0.25))*-1.0)
"""
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Cropping2D, Lambda

model = Sequential()
model.add(Lambda(lambda x: x/255.0,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(16, (5,5), activation='relu', padding='same'))
model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
model.add(Conv2D(48, (5,5), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dropout(0.9))
# model.add(Dense(units=512))
model.add(Dense(units=256))
# model.add(Dense(units=64))
model.add(Dense(units=1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=5,batch_size=32)

model.save('model.h5')
"""