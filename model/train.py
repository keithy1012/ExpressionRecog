import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from datetime import datetime

#CNN Imports
import tensorflow as tf
from tensorflow import keras
from keras import layers, activations, losses, initializers, regularizers, optimizers, metrics
from keras.utils import to_categorical

date_now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

#CONSTANTS
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names=['emotion','pixels','usage']
img_size = 48 # each picture is a list of 48x48 pixels 
EPOCHS = 2
BATCH_SIZE = 40

data = pd.read_csv('fer2013.csv')
print(data.head)

def get_training_data(df):
    x = [] # this will store a list of all the pixels for each image
    train_data = df['pixels'].to_numpy()
    for i in range(len(train_data)):
        x.append(train_data[i].split(' '))
    
    x = np.array(x)
    x = x.astype('float32').reshape(len(train_data), 48, 48, 1)
    return x

X = get_training_data(data) / 255  #divide by 255 to scale every value between [0, 1]
y = data['emotion'].values.astype('int') #converts the emotions to ints
y = to_categorical(y)

#Testing one image:
#plt.imshow(X[0], cmap='gray')
#plt.show()

#Checking for any empty values:
print(data.isnull().sum())

# Checking the distribution of emotions
#sns.countplot(x='emotion', data=data)
#plt.show()

#Begin train/test splitting
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=.25, random_state=42)

#Use 10 images for predicting
x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=10, random_state=42)

#Building a 2D convolution with KERAS
def conv_block(layer_in, n_filters, filter_size=3, padding="same", n_blocks=2, dropout=False):
    for i in range(n_blocks):
        layer_in = layers.Conv2D(n_filters, filter_size, activation="relu", padding=padding)(layer_in)
    layer_in = layers.AveragePooling2D()(layer_in)

    if dropout:
        layer_in = layers.Dropout(.2)(layer_in)
    return layer_in

base = layers.Input(shape=(img_size, img_size, 1)) # Instantiates a Keras Tensor

x = conv_block(base, 32, 3)
x = conv_block(x, 64, 3, )
x = conv_block(x, 128, 3, n_blocks=2, dropout=True)
x = conv_block(x, 256, 3, n_blocks=2, dropout=True)
x = conv_block(x, 512, 3, n_blocks=2, dropout=True)

x=  layers.Flatten()(x) # Flattern makes the multi-dim. tensor into a 1-d
x = layers.Dense(32)(x) #Output is an 'm' dimensional vector --> dense layer are used to change dimensions of vector
x = layers.BatchNormalization()(x) # Essentially normalizes each input to a layer in each mini-batch
x = layers.Activation('relu')(x)

x = layers.Dense(64)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Dense(7, activation='softmax')(x)

model = keras.models.Model(inputs=base, outputs=x)
model.compile(optimizer=optimizers.Adam(.001), loss = losses.categorical_crossentropy, metrics=[metrics.CategoricalAccuracy()])

model.summary()

# Callbacks are objects that can perform actions at various stages of training
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=.5, min_lr=.0001, patience=1, verbose = 1, mode = 'min')
model_ckp = ModelCheckpoint(f'{date_now}.h5', monitor='val_loss', save_best_only=True, mode='min')

history = model.fit(x=x_train, 
                    y=y_train, 
                    epochs=EPOCHS, 
                    batch_size = BATCH_SIZE, 
                    callbacks=[lr_reduce, model_ckp],  # early_stop
                    validation_data=(x_valid, y_valid), 
                    verbose = 1, 
                    steps_per_epoch=len(x_train)//BATCH_SIZE
                   )

# Graphing the losses/accuracy
hist = history.history
plt.plot(hist['loss'], label='loss')
plt.plot(hist['val_loss'], label='val loss')
plt.plot(hist['categorical_accuracy'], label='categorical_accuracy')
plt.plot(hist['val_categorical_accuracy'], label='val categorical_accuracy')
plt.legend()
plt.show()

#Evaluating the Model
model.evaluate(x_valid, y_valid)

#---------------------------------------------------
#Predictions
def predict_img(img):
    img = np.expand_dims(img, axis=0)
    y_pred = model.predict(img)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred

plt.figure(figsize=(12,6))
for i, img in enumerate(x_test):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')

    y_pred = label_map[predict_img(img)[0]]
    
    y_true = label_map[np.argmax(y_test[i])]
    
    plt.title(f'True: {y_true}\nPred: {y_pred}')
    plt.yticks([])
    plt.xticks([])

plt.show()