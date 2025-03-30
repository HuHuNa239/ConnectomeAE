import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.parameter import Parameter

class Rad_AE(nn.Module):
    def __init__(self, n_features, n_lat):
        super().__init__()
        self.encoder = nn.Linear(n_features, n_lat)
        self.decoder = nn.Linear(n_lat, n_features)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_lat),
            nn.Dropout(0.8),
            nn.Linear(n_lat, int(n_lat/2)),
            nn.BatchNorm1d(int(n_lat/2)),
            nn.Dropout(0.8),
            nn.Linear(int(n_lat/2), 500),
            nn.BatchNorm1d(500),
            nn.Dropout(0.7),
            nn.Linear(500, 36),
        )

    def forward(self, x, classify=True):
        x = self.encoder(x)
        x = F.relu(x)
        if classify:
            target = self.classifier(x)
        else:
            target = None
        x = self.decoder(x)
        return x, target

class AE(nn.Module):
    def __init__(self, n_features, n_lat):
        super().__init__()
        self.encoder = nn.Linear(n_features, n_lat)
        self.decoder = nn.Linear(n_lat, n_features)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_lat),
            nn.Dropout(0.8),
            nn.Linear(n_lat, int(n_lat/2)),
            nn.BatchNorm1d(int(n_lat/2)),
            nn.Dropout(0.8),
            nn.Linear(int(n_lat/2), 500),
            nn.BatchNorm1d(500),
            nn.Dropout(0.7),
            nn.Linear(500, 36),
        )

    def forward(self, x, classify=True):
        x = self.encoder(x)
        x1 = F.relu(x)
        if classify:
            target = self.classifier(x1)
        else:
            target = None
        x = self.decoder(x1)
        return x, target

def upper_triangular(adj, nodes):
    upper_tri_mask = torch.triu(torch.ones(nodes, nodes), diagonal=1).bool()
    upper_tri_elements = adj[:, upper_tri_mask]
    return upper_tri_elements

class SC_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.rad_ae = Rad_AE(2250, int(2250 / 2))
        self.ae1 = AE(4005, int(4005 / 2))
        self.ae2 = AE(4005, int(4005 / 2))

        self.mlp_out = nn.Sequential(
            nn.BatchNorm1d(72),
            nn.Dropout(0.6),
            nn.Linear(72, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, s_adj, s_fea, f_adj):
        s_adj = s_adj.to(torch.float32)
        s_fea = s_fea.to(torch.float32)
        f_adj = f_adj.to(torch.float32)
        # # # # # # # # # # # # # # # # # # # # # # #
        fea = torch.flatten(s_fea, start_dim=1)
        fea_restruct, fea_out = self.rad_ae(fea)

        upper = upper_triangular(s_adj, 90)
        f_upper = upper_triangular(f_adj, 90)
        reconstruct, out1 = self.ae1(f_upper)
        r_reconstruct, out2 = self.ae2(reconstruct)
        out = torch.cat([fea_out, out2], dim=1)

        out = self.mlp_out(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0))
        return out, upper, reconstruct, f_upper, r_reconstruct, fea, fea_restruct




