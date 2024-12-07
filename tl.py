import os
import numpy as np
from PIL import Image
import pandas as pd


# put image data into a numpy array
def img_to_np(img_dir):

    images = []

    # loop through images
    for img_file in os.listdir(img_dir)[:100]:

        # save them as numpy arrau
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('L') #grayscale
        img = img.resize((224,224)) #ResNet size
        img = np.array(img).flatten()
        images.append(img)

    return np.array(images)

x_data = img_to_np('images')

# normalize data
x_data = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# get labels for images
img_data = pd.read_csv('Data_Entry_2017_v2020.csv')
y_data = [img_data.loc[img_data['Image Index'] == img, 'Finding Labels'].values[0] 
          for img in os.listdir('images')[:100]]




