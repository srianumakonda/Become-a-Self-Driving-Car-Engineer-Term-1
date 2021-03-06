import cv2
import csv
import tensorflow as tf
import numpy as np

lines = []

with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    image = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
    images.append(image)
    measurements.append(float(line[3]))
    

X_train = np.array(images)
y_train = np.array(measurements)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1024))
model.add(Dense(256))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=5)

model.save('test_model.h5')
