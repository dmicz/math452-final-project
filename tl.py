import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt


from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import tensorflow as tf
from tensorflow import keras




#--------------DATA PROCESSING--------------

# put image data into a numpy array
def preprocess_data(img_dir):

    images = []
    labels = []

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # ResNet input size

    #     # random changes for robustness
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(10),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),

    #     transforms.ToTensor(),         # convert to tensor and normalize [0, 1]
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    # ])

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

        # Normalize the image to the range [0, 1]
        img = tf.cast(img, tf.float32) / 255.0

        # Apply ImageNet normalization: (image - mean) / std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img - mean) / std

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
  

# # Create a dictionary to store one example per label
# label_to_image = {}

# # Iterate through the dataset to find one example per label
# for img, label in zip(x_data, y_data):
#     if label.item() not in label_to_image:  # Add the first occurrence of each label
#         label_to_image[label.item()] = img
#         if len(label_to_image) == num_classes:  # Break once all labels are covered
#             break

# # Plot the images
# fig, axes = plt.subplots(1, len(label_to_image), figsize=(15, 5))
# for ax, (label, img) in zip(axes, label_to_image.items()):
#     img_np = img.permute(1, 2, 0).numpy()  # Convert to HWC format for matplotlib
#     img_np = img_np * 0.229 + 0.485  # Unnormalize using ImageNet stats
#     ax.imshow(img_np.clip(0, 1))  # Ensure values are in valid range
#     ax.set_title(f"Label: {label}")
#     ax.axis("off")

# plt.tight_layout()
# plt.show()


#--------------MODEL--------------

# # load pretrained resnet50
# model = resnet50(weights=ResNet50_Weights.DEFAULT)

# # unfreeze deeper layers for fine-tuning
# for name, param in model.named_parameters():
#     if "fc" in name: 
#         param.requires_grad = True
#     else:
#         param.requires_grad = False


# # modify the final classification layer to match medical data
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 512),  # new fully connected layer
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(512, num_classes)  # final output layer
# )

# # data loaders
# batch_size = 32

# train_dataset = TensorDataset(x_train, y_train)
# val_dataset = TensorDataset(x_val, y_val)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)

# class_weights = compute_class_weight('balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
# class_weights = torch.tensor(class_weights, dtype=torch.float)

# # loss, optimizer, and scheduler for adaptive learning rate
# loss_func = nn.CrossEntropyLoss(weight=class_weights)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
# adapt_lr = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# # train model
# def train_model(model, train_loader, val_loader, loss_func, optimizer, adapt_lr, num_epochs=15):

#     for epoch in range(num_epochs):

#         model.train()
#         train_loss = 0

#         # Training loop
#         for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = loss_func(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item() * inputs.size(0)

#          # validation loop
#         model.eval()
#         val_loss = 0.0
#         correct = 0
#         total = 0

#         with torch.no_grad():
#             for inputs, labels in val_loader:

#                 outputs = model(inputs)
#                 loss = loss_func(outputs, labels)
#                 val_loss += loss.item() * inputs.size(0)

#                 _, preds = torch.max(outputs, 1)
#                 correct += (preds == labels).sum().item()
#                 total += labels.size(0)

#         train_loss = train_loss / len(train_loader.dataset)
#         val_loss = val_loss / len(val_loader.dataset)
#         val_accuracy = correct / total

#         precision = precision_score(labels, preds, average='weighted')
#         recall = recall_score(labels, preds, average='weighted')
#         f1 = f1_score(labels, preds, average='weighted')

#         adapt_lr.step(val_loss)

#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
#         print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


# train_model(model, train_loader, val_loader, loss_func, optimizer, adapt_lr, num_epochs=20)



x_tensor = keras.Input(shape=(224,224,3))
res_net = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=x_tensor)

transfer_learning_model = keras.models.Sequential()
transfer_learning_model.add(res_net)
transfer_learning_model.add(keras.layers.Flatten())
transfer_learning_model.add(keras.layers.Dense(1024, activation='relu'))
transfer_learning_model.add(keras.layers.Dense(num_classes, activation='softmax'))

transfer_learning_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

transfer_learning_model.fit(train_dataset, epochs=25)
print(10*"*", '\nRun Evaluation: ')
transfer_learning_model.evaluate(test_dataset)