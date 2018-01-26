import torch
import torch.nn as nn

# feature size for CUB2011 : 3584
# tf-idf size for CUB2011: 11083


class NetG(nn.Module):
    """
    Feature Generator Network
    outputs synthetic feature for a given tf-idf feature and random noise
    n_tfidf: length of td-idf feature vector
    n_emb: length of text embedding
    n_z: length of the noise vector
    n_feat: length of the final generated feature vector
    """
    def __init__(self, n_tfidf=11083, n_emb=4000, n_z=200, n_mdl=3600, n_feat=3584):
        super(NetG, self).__init__()
        # fully connected layers
        self.fc1 = nn.Linear(in_features=n_tfidf, out_features=n_emb)
        self.fc2 = nn.Linear(in_features=n_emb+n_z, out_features=n_mdl)
        self.fc3 = nn.Linear(in_features=n_mdl, out_features=n_feat)
        # activatio functions
        self.lrelu = nn.LeakyReLU(negative_slope=1e-2)
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(n_emb)
        self.bn2 = nn.BatchNorm2d(n_mdl)

    def forward(self, z, tfidf):
        emb = self.bn1(self.fc1(tfidf))
        x = torch.cat([z, emb], 1)
        x = self.bn2(self.lrelu(self.fc2(x)))
        x = self.tanh(self.fc3(x))
        return x


class NetD(nn.Module):
    """
    Feature Discriminator Network
    """
    def __init__(self, n_feat=3584, n_mdl=2000, num_cls=200):
        super(NetD, self).__init__()

        self.fc1 = nn.Linear(in_features=n_feat, out_features=n_mdl)
        self.fc_s = nn.Linear(in_features=n_mdl, out_features=1)
        self.fc_cls = nn.Linear(in_features=n_mdl, out_features=num_cls)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.bn1 = nn.BatchNorm1d(n_mdl)

    def forward(self, x):
        x = self.bn1(self.relu(self.fc1(x)))
        s = self.sigmoid(self.fc_s(x))
        cl = self.log_softmax(self.fc_cls(x))
        return s, cl


'''
class VPDE(nn.Module):
    """
    Feature Extractor Network
    """
    def __init__(self):
        super(self, VPDE).__init__()
'''