from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.datasets
import torchvision.transforms 
from datetime import datetime
from model import Siamese_Network
import os, sys
from torch.utils.data import DataLoader, Dataset


import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image
import PIL.ImageOps    
from torch.utils.data import Dataset



# NEED TO ADD IN PARAMETERS SO THE OPTIMIZER WORKS AS INTENDED 
# ADD CONVOLUTIONAL LAYERS AND LINEAR LAYERS 
# BUT ALSO TRY TO FORMAT THE DATA INTO PAIRS LIKE WE DISCUSSED 
# POSITIVE: (ALTERED UNET OUTPUT (ANCHOR), TARGET (SAME)) AND 
# NEGATIVE: (ALTERED UNET OUTPUT (ANCHOR), TARGET (DIFFERENT))

# Showing images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

# Plotting data
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def train(model, device, train_loader, optimizer, epoch, criterion):
    counter = []
    loss_history = []
    iteration_number = 0

    for i, (img0, img1, label) in enumerate(train_loader, 0):

        # Send the images and labels to CUDA
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

        # Zero the gradients
        optimizer.zero_grad()

        # Pass in the two images into the network and obtain two outputs
        output1, output2 = model(img0, img1)

        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = criterion(output1, output2, label)

        # Calculate the backpropagation
        loss_contrastive.backward()

        # Optimize
        optimizer.step()

        # Every 10 batches print out the loss
        if i % 10 == 0 :
            print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

    show_plot(counter, loss_history)

def test():
    return


# need to create custom dataset so Pytorch can convert bmp files to tensors
# in order to use the torchvision.dataset functionalities
class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_dist = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_dist, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2))
        return loss_contrastive

def main():
    
    directory_train_images = datasets.ImageFolder(root='../../train_images')
    transform_train = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()
                                    ])
    
    transform_test = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()
                                    ])
        
    print("BMP Train set: ", directory_train_images)
    trainset = SiameseNetworkDataset(directory_train_images, transform_train) 
    
    print("LEN: ", len(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = Siamese_Network().to(device)
        
    criterion = ContrastiveLoss()

    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    # Create a simple dataloader just for simple visualization
    vis_dataloader = DataLoader(trainset,
                            shuffle=True,
                            num_workers=2,
                            batch_size=8)

    # Extract one batch
    example_batch = next(iter(vis_dataloader))

    # Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
    # If the label is 1, it means that it is not the same person, label is 0, same person in both images
    concatenated = torch.cat((example_batch[0], example_batch[1]),0)

    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy().reshape(-1))
    # for epoch in range(20):
    #         train(model, device, train_loader, optimizer, epoch, criterion)
        

if __name__ == "__main__":
    main()