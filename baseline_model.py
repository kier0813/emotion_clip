import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import cv2
import glob
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import os

device = torch.device("cuda")


def resize_image(input_path, output_path):
    
    # Open the image
    image = Image.open(input_path)

    # Usable format for CLIP-Dissect is 224 x 224
    new_size = (224, 224)

    # Resize the image
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Save the resized image
    resized_image.save(output_path)



def resize_images_in_folder(input_folder, output_folder):
    
    for root, dirs, files in os.walk(input_folder):
        
        for file in files:
            
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                
                # Get input and output paths for the current image
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))

                # Create the output folder structure if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Resize the image and save it to the new structure
                resize_image(input_path, output_path)


class FaceEmotionClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        images= images.to(device)
        labels= labels.to(device) 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        images= images.to(device)
        labels= labels.to(device) 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}],  train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class Deep_Emotion(FaceEmotionClassificationBase):
    def __init__(self):

        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(3,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,7)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def attention(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,input):
        out = self.attention(input)

        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Deep_Emotion().to(device)

def evaluate(model,val_loader):
    model.eval()
    outputs =[model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

import random

class Coin:
    @staticmethod
    def toss():
        # Simulates a coin toss, returns 0 or 1
        return random.randint(0, 1)

class Dice:
    @staticmethod
    def toss():
        while True:
            # Generate a 3-bit binary number using coin tosses
            num = (Coin.toss() << 2) + (Coin.toss() << 1) + Coin.toss()
            
            # Convert the binary number to decimal (1-6)
            result = num + 1

            # Check if the result is within the valid range
            if 1 <= result <= 6:
                return result
            
import numpy as np

class NormBase:
    """Base class of normalizations."""
    def __init__(self, num_channels, gamma=None, beta=None, eps=1e-5):
        """Initialization."""
        # gamma and beta are learnable parameters which are denoted as gamma and beta in the equation.
        self.C = num_channels
        self.eps = eps

        self.gamma = gamma if gamma is not None else np.random.uniform(size=self.C)
        self.beta = beta if beta is not None else np.random.uniform(size=self.C)

    def forward(self, input):
        raise NotImplementedError

class InstanceNorm(NormBase):
    def forward(self, input):
        # Implement the forward pass for instance normalization
        N, C, H, W = input.shape
        normalized_input = np.zeros_like(input)
        
        for n in range(N):
            for c in range(C):
                mean = np.mean(input[n, c, :, :])
                std = np.std(input[n, c, :, :])
                normalized_input[n, c, :, :] = (input[n, c, :, :] - mean) / np.sqrt(std ** 2 + self.eps)
                normalized_input[n, c, :, :] = self.gamma[c] * normalized_input[n, c, :, :] + self.beta[c]
                
        return normalized_input


def fit(num_epochs, model, train_loader,val_loader,opt=optimizer):
    history=[]
    for epoch in range(num_epochs):
        model.train()
        train_losses=[]
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        print("Epoch-{},Loss-{}".format(epoch,loss.item()))

        result = evaluate(model,val_loader)
        result['train_loss'] = sum(train_losses)/len(train_losses)
        model.epoch_end(epoch,result)
        history.append(result)
    
    return history
