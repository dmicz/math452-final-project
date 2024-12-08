import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# put image data into a numpy array
def preprocess_data(img_dir, num_img):

    images = []

    # loop through images
    for img_file in os.listdir(img_dir)[:num_img]:

        # save them as numpy arrau
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('RGB') # 3 channels to match ImageNet
        img = img.resize((224,224)) #ResNet size
        img = np.array(img) / 255
        images.append(img)

    # get labels for images
    img_data = pd.read_csv('Data_Entry_2017_v2020.csv')
    y_data = [img_data.loc[img_data['Image Index'] == img, 'Finding Labels'].values[0] 
            for img in os.listdir('images')[:num_img]]

    # encode labels numericaly
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(y_data)

    return np.array(images), y_data

x_data, y_data = preprocess_data('images', 100)

# normalize to fit ImageNet mean and sd
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
x_data = (x_data - np.array(imagenet_mean).reshape(1, 1, 1, 3)) / np.array(imagenet_std).reshape(1, 1, 1, 3)

# split into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)