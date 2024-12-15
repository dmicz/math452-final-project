import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras

#--------------DATA PROCESSING--------------

# put image data into a numpy array
def preprocess_data(img_dir):

    images = []
    labels = []

    img_data = pd.read_csv('Data_Entry_2017_v2020.csv')

    # loop through images
    i = 1
    for img_file in tqdm(os.listdir(img_dir), desc=f"loading images"):

        # save them as numpy arrau
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('RGB') # 3 channels to match ImageNet


        img = tf.image.resize(img, [224, 224])

        img = tf.image.random_contrast(img, lower=0.8, upper=1.2) 
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_flip_left_right(img)

        # normalize the image to the range [0, 1]
        img = tf.cast(img, tf.float32) / 255.0

        # apply ImageNet normalization: (image - mean) / std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img - mean) / std

        # add random noise
        noise = tf.random.uniform(shape=tf.shape(img), minval=0, maxval=1, dtype=tf.float32)
        img = tf.where(noise < 0.05 / 2, 0.0, tf.where(noise < 0.05, 1.0, img))

        images.append(img)

        # get label
        label = img_data.loc[img_data['Image Index'] == img_file, 'Finding Labels'].values[0]
        labels.append(label)

        i += 1

    # encode labels )numericaly
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # stack images into a tensor
    # images = torch.stack(images) 
    # labels = torch.tensor(labels, dtype=torch.long)

    # images = tf.convert_to_tensor(images, dtype=tf.float32) 
    # labels = tf.convert_to_tensor(labels, dtype=tf.int32) 

    return np.array(images), np.array(labels)


images, labels = preprocess_data('images')
num_classes = len(set(labels))

# split data into training and testing sets (e.g., 80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

x_train = tf.convert_to_tensor(x_train)
x_test = tf.convert_to_tensor(x_test)

y_train = tf.convert_to_tensor(y_train)
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.convert_to_tensor(y_test)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

# Convert them to TensorFlow datasets (for batching, shuffling, etc.)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Shuffle, batch, and prefetch the datasets for performance
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

x_tensor = keras.Input(shape=(224,224,3))
res_net = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=x_tensor)

for layer in res_net.layers:
    res_net.trainable = True

transfer_learning_model = keras.models.Sequential()
transfer_learning_model.add(res_net)
transfer_learning_model.add(keras.layers.Flatten())
transfer_learning_model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
transfer_learning_model.add(keras.layers.Dropout(0.25))
transfer_learning_model.add(keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01)))
transfer_learning_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics = \
                                ['accuracy', tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

transfer_learning_model.fit(train_dataset, epochs=30)
print(10*"*", '\nRun Evaluation: ')
transfer_learning_model.evaluate(test_dataset) 