from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from DataGenerator import DataGenerator
import numpy as np

# ----------------------------------------------------------------
# Parameters
n_classes = 20
n_epochs = 5
batch_size = 50
img_width, img_height, img_channel = 299, 299, 3

params = {'n_classes': n_classes,
          'dim_x': img_width,
          'dim_y': img_height,
          'dim_z': img_channel,
          'batch_size': batch_size,
          'shuffle': True}

dataFolder = 'E:\\ExampleDatasetFolder\\'
fileFolder = 'E:\\ExampleFileFolder\\'
trainDataPath = fileFolder + 'train-data.txt'
trainLabelPath = fileFolder + 'train-label.txt'
testDataPath = fileFolder + 'test-data.txt'
testLabelPath = fileFolder + 'test-label.txt'

trainLabel = np.loadtxt(trainLabelPath)
testLabel = np.loadtxt(testLabelPath)

trainDataFileList = []
file1 = open(trainDataPath)
nextLine = file1.readline()
while nextLine != "":
    trainDataFileList.append(dataFolder + nextLine[:-1])
    nextLine = file1.readline()

testDataFileList = []
file2 = open(testDataPath)
nextLine = file2.readline()
while nextLine != "":
    testDataFileList.append(dataFolder + nextLine[:-1])
    nextLine = file2.readline()

partition ={'train': trainDataFileList, 'validation': testDataFileList}
labels = {}
for i in range(len(trainLabel)):
    labels[trainDataFileList[i]] = trainLabel[i]
for i in range(len(testLabel)):
    labels[testDataFileList[i]] = testLabel[i]

print(trainLabel.shape)
print(trainDataFileList.__len__())
print(testLabel.shape)
print(testDataFileList.__len__())

# Generators
training_generator = DataGenerator(**params).generate(labels, partition['train'])
validation_generator = DataGenerator(**params).generate(labels, partition['validation'])

# ----------------------------------------------------------------
# Design model

# ---------------------------------------
# ----- Your Network Architecture -------
# ---------------------------------------

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# ----------------------------------------------------------------
# Train model on dataset
model.fit_generator(generator = training_generator,
                    steps_per_epoch = len(partition['train'])//batch_size,
                    validation_data = validation_generator,
                    validation_steps=len(partition['validation'])//batch_size,
                    epochs=n_epochs)

