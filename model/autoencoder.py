import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim=773, latent_dim=50):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(True),
            nn.Linear(400, latent_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(True),
            nn.Linear(400, input_dim),
            nn.Sigmoid()  # 使用Sigmoid保证输出值在[0,1]范围，根据实际情况选择激活函数
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

from sklearn.cluster import KMeans

def cluster_features(encoded_features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(encoded_features)
    return kmeans.labels_, kmeans.cluster_centers_