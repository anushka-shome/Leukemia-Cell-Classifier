import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import _testimportmultiple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import timm
from tqdm import tqdm
from CellClassifier import CellClassifier
#The dataset will be a folder in the directory. Each image will go in corresponding folders based on their classification

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    #property
    def classes(self):
        return self.data.classes
    
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), ])
#All images are the same size after transform is applied
data_dir = '/Users/anushkashome/StreamLit/test/training_data'
dataset = MyDataset(data_dir, transform)
#len(dataset)
#dataset[5]
#Will contain the image and class number
#image, label = dataset[5]
#To get the class associated with each class number
target = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
print(target)

#transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), ])
print(len(dataset))
image, label = dataset[50]
image # Worry about image showing up later
print(label)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#Dataset batched into 32 examples
for images, labels in dataloader:
    break

print(labels)

#Model
'''
class CellClassifier(nn.Module):
    def __init__(self, num_classes=2): #Define parts of model
        super(CellClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x): #connect parts
        x = self.features(x)
        output = self.classifier(x)
        return output
'''
model = CellClassifier(num_classes=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("mps")
else:
    print("Still using cpu")
'''
model.to(device)
#print((model(images)).shape)
example_out = model(images)

#training
#Loss function
criterion = nn.CrossEntropyLoss()
#Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(criterion(example_out, labels))

#test_folder = "/Users/anushkashome/StreamLit/test/testing_data/C-NMC_test_final_phase_data"
valid_folder = "/Users/anushkashome/StreamLit/test/validation_data"
#data_dir has train_folder

train_dataset = MyDataset(data_dir, transform=transform)
val_dataset = MyDataset(valid_folder, transform=transform)
#test_dataset = MyDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#CPU takes too long for training, need to run on GPU -- put above
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_epoch = 5 #iteration of data set
train_losses, val_losses = [], []

for epoch in range(num_epoch):
    model.train()
    print("trained")
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    i = 0
    for images, labels in train_loader:
        i = i + 1
        print("train here", i)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() #back propogation, updating weights
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    print("after first for")
    train_loss = running_loss/len(train_loader.dataset)
    train_losses.append(train_loss)

    #Validation
    model.eval()
    print("eval")
    running_loss = 0.0
    j = 0
    with torch.no_grad():
        for images, labels in val_loader:
            j = j + 1
            print("val here", j)
            images,labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss = loss.item() * images.size(0)
    val_loss =  running_loss/len(val_loader.dataset)
    val_losses.append(val_loss)
    print("after second for loop")
    progress_bar.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1}/{num_epoch} - Train loss:{train_loss}, Validation loss{val_loss}")

torch.save(model.state_dict(), 'model.pth')


