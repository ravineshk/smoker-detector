

import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import imutils 
from imutils import paths 
import sklearn 
import os 

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

dataset = 'dataset'
imagePaths = list(paths.list_images(dataset))

#Read all the images and assign them to the labels 
data=[]
labels = []

for i in imagePaths:
    #Extract the labels from the file names
    label = i.split(os.path.sep)[-2]  
    #Load the input images and preprocess it 
    image = load_img(i,target_size=(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)
    #Update the data and label lists
    data.append(image)
    labels.append(label)

#Convert the data and labels to numpy array 
data = np.array(data, dtype= "float32")
labels = np.array(labels)

#Convert the labels 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#Partition the data into train and test 
(trainX,testX,trainY,testY) = train_test_split(data,labels, test_size = 0.20,stratify=labels,random_state=10)

#Visualizing the data 
print('Number of images in the training set   : ', len(trainX))
print('Number of images in the validation set : ', len(testY))

#Construct the training image generator for data augmentation 
aug = ImageDataGenerator(
                         rotation_range = 40,
                         zoom_range = 0.25,
                         width_shift_range = 0.2,
                         height_shift_range = 0.2,
                         shear_range = 0.2,
                         horizontal_flip = True,
                         fill_mode ='nearest')

#Load the pre trained model 
baseModel = MobileNetV2(weights='imagenet', include_top = False, input_shape =(224,224,3))

#Constructing the head of the model that will be placed on the top of the base model 
headModel = baseModel.output
headModel = AveragePooling2D(pool_size = (7,7))(headModel)
headModel = Flatten(name = 'flatten')(headModel)
headModel = Dense(128, activation = 'relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = 'softmax')(headModel)

#Place the headModel on the top of the baseModel 
model = Model(inputs= baseModel.input, outputs = headModel)

#Freeze the layers of base model so they will not be update 
for layer in baseModel.layers:
    layer.trainable = False 

INIT_LR = 0.0001
EPOCHS= 30
BATCH_SIZE = 32

print('The batch_size of the images : ', BATCH_SIZE)
print('The epoch value for the training the model : ', EPOCHS)

#Compile the Model
opt = Adam(learning_rate = INIT_LR, decay = INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer = opt, metrics=['accuracy'])

#Train the Model
H = model.fit(
              aug.flow(trainX, trainY, batch_size = BATCH_SIZE),
              steps_per_epoch = len(trainX)//BATCH_SIZE,
              validation_data = (testX,testY),
              validation_steps = len(testX)//BATCH_SIZE,
              epochs = EPOCHS,
              )

#Plot the training results 
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']

loss = H.history['loss']
val_loss = H.history['val_loss']

epochs_range = range(30)
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label='Training Accuracy')
plt.plot(epochs_range,val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,loss, label='Training Loss')
plt.plot(epochs_range,val_loss,label='Validation Loss')
plt.legend(loc ='upper right')
plt.title('Training and Validation Loss')

#Predict the results 
predict = model.predict(testX,batch_size = BATCH_SIZE)
predict = np.argmax(predict,axis=1)
print(classification_report(testY.argmax(axis=1),predict,target_names= lb.classes_))

#To save the model 
model.save('mobilenetv2_face_mask_detector_dataset3.h5')

