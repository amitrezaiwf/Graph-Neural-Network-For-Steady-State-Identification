import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from random import shuffle
from torch.nn import Linear
import torch.nn.functional as F
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import *
from user_utility_functions import * 
import seaborn as sns

#-
torch.manual_seed(12275)
epoch = 200
hidden_channels = 64 # Number of layers
b_size = 64 # Batch size
num_features = 5

################## Training Data loading #####################
train_data = torch.load('train_dataset_sample.pt', map_location = torch.device("cpu"), weights_only = False)
random.shuffle(train_data)
train_loader = DataLoader(train_data, batch_size = b_size, shuffle = True)

################## Model parameter setting #####################
learning_rate = 1e-5 #0.01
weight_decay = 5e-4

#while True:
print("\n Main Menu:")
print("1. GCN")
print("2. GAT")
print("3. Graph Transformer")
print("0. Exit")

choice = input("Enter your choice: ")

if choice == "1":
    print("You selected GCN")
    model = GCN(hidden_channels = b_size)
elif choice == "2":
    print("You selected GAT")
    model = GAT(hidden_channels = b_size)
elif choice == "3":
    print("You selected Graph Transformer")
    model = GraphTranformer(hidden_channels = b_size)
elif choice == "0":
    print("Exiting...")
    exit(0)
else:
    print(" Invalid choice! Please enter a valid option.")

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 5e-4)
criterion = torch.nn.MSELoss()

def train(train_loader):
    model.train()
    original_train = []
    train_label_data = []
    x_train = 0
    predicted_labels = []
    graph_nodes = []
    pred_ipr = []
    actual_ipr = []

    for data in train_loader:  # Iterate in batches over the training dataset.
        out_train = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.

        train_labels, graph_size, pred_labels = decides_class_batches(data, out_train)

        predicted_labels.extend(pred_labels.detach().numpy().tolist())
        train_label_data.extend(train_labels.detach().numpy().tolist())
        graph_nodes.extend(graph_size.detach().numpy().tolist())
        original_train.extend(data.y.detach().numpy().tolist())
        out_train = out_train.flatten()
        
        # For regression purpose
        actual_ipr.extend(data.y.detach().numpy().tolist())
        pred_ipr.extend(out_train.detach().numpy().tolist())

        ################# FOR GRAPH TRNASFORMER  ################### 
        if choice == "3":
           out_train = out_train.to(torch.float32)
           data.y = data.y.to(torch.float32)

        train_loss = criterion(torch.log(1 + out_train), torch.log(1 + data.y))
        train_loss = train_loss.float()
        
        train_loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        x_train += train_loss.item()
    
    return x_train/len(train_loader), graph_nodes, original_train, train_label_data, predicted_labels, actual_ipr, pred_ipr

##################### Training ###########################

train_losses = []
train_accuracy = []
test_losses = []
test_accuracy = []
epochs = []

# KDE plot: Define colors for the initial, intermediate, and final plots
initial_color = '#09b976'
intermediate_color = '#D3D3D3' #'gray'
final_color = '#D10C60'

# Initialize lists to hold flattened weights for each weight matrix across epochs
all_weight_gc1 = []
all_weight_gc2 = []
all_weight_gc3 = []

img_path = "C:\\Users\\THINKPAD\\Dropbox\\Priodyuti\\Bar-Ilan\\Amit\\LOC-CODE\\Figure\\heatmaps\\"

# Step 3: Create a KDE plot
for i in range(1, epoch + 1):
    train_loss, graph_nodes, original_train, train_label_data, predicted_labels, actual_ipr, pred_ipr = train(train_loader)
    train_losses.append(train_loss) #.detach().numpy().tolist())
    print(i,"Train Loss: ",train_loss) #.detach().numpy().tolist())
    epochs.append(i)

# Save the model's state dictionary
torch.save(model.state_dict(), 'GCN_model.pth')

############ Epoch vs. Loss ########################
plt.figure(figsize = (10, 6))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epochs, train_losses, linewidth = 2)
plt.show()

################## Training time actual vs. Predicted ########################
plt.figure(figsize=(10, 6))
plt.plot(actual_ipr,actual_ipr,color = '#5B2C6F', marker = 'o',markersize = 16, linestyle = 'None', markerfacecolor = "none")
plt.plot(actual_ipr,pred_ipr,color = '#DE3163', marker = "^", markersize = 8, linestyle = 'None', markerfacecolor = "none")
plt.legend(["actual", "predicted"], loc = "lower right")
plt.show() 


print('Completion of training process...\n') 
