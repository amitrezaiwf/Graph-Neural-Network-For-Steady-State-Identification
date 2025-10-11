import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

import random
from random import shuffle
from model import *
from user_utility_functions import *
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#-
torch.manual_seed(12275)
b_size = 64
hc = 64
num_features = 5
####################### LOAD TEST DATASETS  ##########################################
test_data = torch.load('test_dataset_samples.pt', map_location = torch.device("cpu"), weights_only = False)
random.shuffle(test_data)
test_loader = DataLoader(test_data, batch_size = b_size, shuffle = False)

########################### LOAD MODEL  #####################################

# Create a new instance of the model
load_model = GCN(hidden_channels = hc)

# Load the state dictionary into the new model instance
load_model.load_state_dict(torch.load('GCN_model.pth'))

criterion = torch.nn.MSELoss()

# Select a graph from the dataset to analyze
def test(test_loader):
    load_model.eval()
    x_test = 0
    original_test = []
    test_pred = []
    test_label_data = []
    test_pred_labels = []
    actual_test_ipr = []
    pred_test_ipr = []
    with torch.no_grad():
     for data in test_loader:  # Iterate in batches over the training/test dataset.
        out_test = load_model(data.x, data.edge_index, data.batch)
        
        test_labels, graph_size, pred_labels = decides_class_batches(data, out_test)
        
        original_test.extend(data.y.detach().numpy().tolist())
        test_pred.extend(out_test[:,0].detach().numpy().tolist())
       
        test_label_data.extend(test_labels.detach().numpy().tolist())
        test_pred_labels.extend(pred_labels.detach().numpy().tolist())

        out_test = out_test.flatten()
        
        actual_test_ipr.extend(data.y.detach().numpy().tolist())
        pred_test_ipr.extend(out_test.detach().numpy().tolist())
    
        test_loss = criterion(torch.log(1 + out_test), torch.log(1 + data.y)) 
        x_test += test_loss.item()

    return x_test/len(test_loader), original_test, test_pred, test_label_data, test_pred_labels, actual_test_ipr, pred_test_ipr

test_Loss, original_test, test_pred, test_label_data, test_pred_labels, actual_test_ipr, pred_test_ipr  = test(test_loader)

plt.figure(figsize = (6, 4))
plt.plot(actual_test_ipr,actual_test_ipr,color = '#DE3163', marker = "s", markersize = 8, linestyle = 'None', markerfacecolor = "none")
plt.plot(actual_test_ipr,pred_test_ipr,color = '#229954', marker = "^", markersize = 8, linestyle = 'None', markerfacecolor = "none")
plt.legend(["test-actual", "test-predicted"], loc = "lower right")
plt.xlabel('True')
plt.ylabel('Predicted')
plt.rcParams.update({'font.size': 22})
plt.savefig('test_actual_vs_prediction.png', dpi = 300)
plt.show()


err1 = (np.array(actual_test_ipr) - np.array(pred_test_ipr))/(np.array(actual_test_ipr))
err2 = np.array(pred_test_ipr)/np.array(actual_test_ipr)
bins = 20

plt.figure(figsize = (6, 4))
plt.hist((np.array(actual_test_ipr) - np.array(pred_test_ipr)), bins = bins)
plt.savefig('abs_err.png', dpi = 300)

plt.figure(figsize = (6, 4))
plt.hist(err1*100, bins = bins)
plt.savefig('per_err.png', dpi = 300)

plt.figure(figsize = (6, 4))
plt.hist(err2, bins = bins)
plt.savefig('ratio_err.png', dpi = 300)

############################# Binary Classification ########################################
actual_test_class, pred_test_class = decides_predicted_classes(actual_test_ipr, pred_test_ipr)

cm = sklearn.metrics.confusion_matrix(actual_test_class, pred_test_class)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.show()

############# Convert the confusion matrix to percentages
cm_percentage = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis] * 100

# Plotting the confusion matrix with percentages
plt.figure(figsize = (8, 6))
sns.heatmap(cm_percentage, annot = True, fmt = ".0f", cmap = "crest", cbar = True)

# Adjust the color bar scale
cbar = plt.gca().collections[0].colorbar
cbar.set_label('Percentage')

# Set labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix', dpi = 300)

######################################## Multiclass confusion matrix #############################################
actual_test_multiclass, pred_test_multiclass = decides_predicted_multi_classes(actual_test_ipr, pred_test_ipr)

cm = sklearn.metrics.confusion_matrix(actual_test_multiclass, pred_test_multiclass) #, fmt = '.2%', cmap = 'Blues')
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.savefig('confusion_matrix_normal_multiclass', dpi = 300)

############# Convert the confusion matrix to percentages
cm_percentage = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis] * 100

# Plotting the confusion matrix with percentages
plt.figure(figsize = (8, 6))
sns.heatmap(cm_percentage, annot = True, fmt = ".0f", cmap = "crest", cbar = True) # ,cmap='Blues'

# Adjust the color bar scale
cbar = plt.gca().collections[0].colorbar
cbar.set_label('Percentage')

# Set labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_multiclass', dpi = 300)
