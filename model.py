import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GINConv
from torch_geometric.data import InMemoryDataset

num_features = 5

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(num_features, hidden_channels, heads = 4, concat = True)
        self.conv2 = GATConv(hidden_channels*4, hidden_channels, heads = 1, concat = False)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p = 0.6, training = self.training)
        x = self.lin(x)
        
        return x.double()

class GraphTranformer(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GraphTranformer, self).__init__()
        self.conv1 = TransformerConv(num_features, hidden_channels, heads = 2)
        self.conv2 = TransformerConv(hidden_channels * 2, hidden_channels, heads = 1)
        self.lin = Linear(hidden_channels, 1)  # Output layer for regression
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)  # Pooling to get graph-level output
        x = F.dropout(x, p = 0.6, training = self.training)
        x = self.lin(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)
       
    def forward(self, x, edge_index, batch):
        
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p = 0.6, training = self.training)
        x = self.lin(x)
        
        return x.double()
