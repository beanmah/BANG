
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNEncoder, self).__init__()

        self.conv1 = GINConv(Sequential(
            Linear(input_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        ))

        self.conv2 = GINConv(Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        ))

        self.conv3 = GINConv(Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        ))

        self.conv4 = GINConv(Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        ))

        self.conv5 = GINConv(Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        ))

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)

        x = global_add_pool(x, batch)

        return x

class GNNClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super(GNNClassifier, self).__init__()
        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index, batch)
        out = self.classifier(x)
        return out
