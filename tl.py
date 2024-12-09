import os
import torch
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision import transforms

from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

#--------------DATA PROCESSING--------------

# put image data into a numpy array
def preprocess_data(img_dir, num_img):

    images = []
    labels = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),         # convert to tensor and normalize [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    img_data = pd.read_csv('Data_Entry_2017_v2020.csv')

    # loop through images
    i = 1
    for img_file in tqdm(os.listdir(img_dir)[:num_img+1], desc=f"loading images"):

        # save them as numpy arrau
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('RGB') # 3 channels to match ImageNet
        img = transform(img)
        images.append(img)

        # get label
        label = img_data.loc[img_data['Image Index'] == img_file, 'Finding Labels'].values[0]
        labels.append(label)

        i += 1

    # encode labels numericaly
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # stack images into a tensor
    images = torch.stack(images) 
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels

x_data, y_data = preprocess_data('images', 5000)

# split into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
num_classes = len(set(y_data.numpy()))  

#--------------MODEL--------------

# load pretrained resnet50
model = resnet50(pretrained=True)


# modify the final classification layer to match medical data
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),  # new fully connected layer
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)  # Final output layer
)

# data loaders
batch_size = 16

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# loss, optimizer, and scheduler for adaptive learning rate
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
adapt_lr = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# train model
def train_model(model, train_loader, val_loader, loss_func, optimizer, adapt_lr, num_epochs=10):

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0

        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        adapt_lr.step()

         # validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:

                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

train_model(model, train_loader, val_loader, loss_func, optimizer, adapt_lr, num_epochs=10)