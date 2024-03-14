import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GCN1(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=512, dr=0.6): 
        super().__init__()
        self.dr = dr

        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)
        self.bn3 = BatchNorm(num_classes)

    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.bn1(self.conv1(x, edge_index))
        x2 = F.relu(x1)
        x3 = F.dropout(x2, p=self.dr, training=self.training)
        x4 = self.bn2(self.conv2(x3, edge_index))
        x5 = F.relu(x4)
        x6 = F.dropout(x5, p=self.dr, training=self.training)
        out = self.bn3(self.conv3(x6, edge_index))

        return out, x6 


