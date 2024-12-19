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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

        # resize for same shape as ImageNet
        img = tf.image.resize(img, [224, 224])

        # random data augmentation
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

# convert to tf datasets 
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_dataset = train_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

# set up pre-trained models
x_tensor = keras.Input(shape=(224,224,3))
pre_trained = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=x_tensor)

# define callbacks
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='rn_trying.keras', monitor='val_accuracy',\
                                                      save_best_only=True,mode='max',verbose=1)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# allow fine tuning on the last 10 layers
for layer in pre_trained.layers[:-10]:
    pre_trained.trainable = True

# add layers to the model
transfer_learning_model = keras.models.Sequential()
transfer_learning_model.add(pre_trained)
transfer_learning_model.add(keras.layers.Flatten())
transfer_learning_model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005)))
transfer_learning_model.add(keras.layers.Dropout(0.5))
transfer_learning_model.add(keras.layers.BatchNormalization())
transfer_learning_model.add(keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.005)))
transfer_learning_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics = \
                                ['accuracy',tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

# train model
transfer_learning_model.fit(train_dataset, validation_data=test_dataset, epochs=50, callbacks=[checkpoint_callback, lr_scheduler, early_stopping])

# evaluate model
model = keras.models.load_model('rn_trying.keras')
model.evaluate(test_dataset)

# plot confusion matrix
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title(f'confusion matrix')
plt.show()