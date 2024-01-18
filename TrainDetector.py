#Libraries
import cv2 #OpenCV library open-source library that can be used to perform tasks like face detection, objection tracking, landmark detection
from keras.models import Sequential #Sequential Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten #Layers
from keras.optimizers import Adam #optimizer for updating weights during training
from keras.preprocessing.image import ImageDataGenerator #utility for preproccessing
import os #python module for OS interaction

#Data preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #To suppress unecessary warnings
#creating ImageDataGenerator instances for training and validation data
train_data_gen = ImageDataGenerator(rescale=1. / 255)
validation_data_gen = ImageDataGenerator(rescale=1. / 255)

# Here we are loading training and testing data from each of their respective directories to create their respective generators 
# use of generator --> to efficiently load and preprocess data for training and evaluation during the training process of a model
# Images are resized to (48, 48) pixels, converted to grayscale this is done so that training of data can be done faster
train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')


validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

#CNN model Formation
emotion_model = Sequential()

#Layer info -->  there are a total of 13 layers in the neural network model.

#first convolutional layer. It applies 32 filters of size 3x3 to the input images, using the ReLU activation function. It processes grayscale images of size 48x48 pixels.
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
#second convolutional layer with 64 filters of size 3x3 learns more complex features compared to the first layer.
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

#maxpooling is downsizing the image while retaining the orignal features of the image and it is done so that image processing can be done faster as performing feature extraction and processing on the orignal image size will increase the training time of the model a lot.it resutls in a 2x2 matrix with all features intact
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

#Dropout sets a fraction of input units to zero during training. It helps prevent overfitting
emotion_model.add(Dropout(0.25))

#Increasing the number of filters allows the model to capture more complex patterns in the data.
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

#Flattens the output maxtrix from above to a one-dimensional array. This prepares the data for the fully connected layers.
emotion_model.add(Flatten())
#Adding fully connected layers
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

#Compiling of model
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

#training of model --> trained using the fit_generator method.It trains for 50 epochs with the training and validation data generators.
emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=7178 // 64)

#Saving the model in json and H5 file.
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

emotion_model.save_weights('emotion_model.h5')
