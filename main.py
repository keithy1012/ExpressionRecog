from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os
import csv 
from datetime import datetime 
import matplotlib.pyplot as plt

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model("model/13-06-2023-16-31-15.h5")

#Predictions
def predict_img(img):
    img = np.expand_dims(img, axis=0)
    y_pred = model.predict(img)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)
    return y_pred


def img_to_array(img):
    #Image to numpy array
    image = Image.open(img).resize((48, 48)).convert('L') #resize to 48x48 and convert to grayscale
    arr = np.array(image)
    plt.imshow(arr, cmap='gray')
    plt.show()
    print(type(arr))
    return arr


if __name__ == "__main__":
    directory = "test_data"
    for filename in os.listdir(directory):
        print(filename)
        date_time = filename[0:-4]
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and (f.endswith('.png') or f.endswith('.jpg')):
            image_array = img_to_array(f)
            datetime_format = '%d-%m-%Y-%H-%M-%S'
            datetime = datetime.strptime(date_time, datetime_format)
            date = str(datetime.month) + "/" + str(datetime.day) + "/" + str(datetime.year)
            time = str(datetime.hour) + ":" + str(datetime.minute)
  
            result = [image_array, label_map[predict_img(image_array)[0]], date, time]
            with open ('result.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(result)
