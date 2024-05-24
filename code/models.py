import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import sklearn
import sklearn.cluster
import math
import torch

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    '''
    2-layer GCN with dropout
    '''
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
def cluster(data, k, num_iter, init, cluster_temp=5):
    '''
    pytorch (differentiable) implementation of soft k-means clustering.
    Input: 
        data: embeddings
        k: number of clusters
        init: initial cluster means
        cluster_temp
    Output:
        mu: cluster means
        r: soft assignment
        dist: distance between nodes and cluster center
    ''' 
    #normalize x 
    data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
    # initialize clusters
    mu = init
    
    for t in range(num_iter):
        dist = data @ mu.t()
        r = torch.softmax(cluster_temp*dist, 1)
        cluster_r = r.sum(dim=0)
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        # print(cluster_r)
        new_mu = torch.diag(1/cluster_r) @ cluster_mean
        mu = new_mu
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp*dist, 1)
    return mu, r, dist

class GCNClusterNet(nn.Module):
    '''
    The ClusterNet architecture. 
    2-layer GCN produce embeddings
    Input: edges and matrix which describes features of nodes
    
    r: soft clustering assignment
    dist: node similarities
    '''
    def __init__(self, no_features, no_hidden, no_output, dropout, K, cluster_temp):
        super(GCNClusterNet, self).__init__()

        self.GCN = GCN(no_features, no_hidden, no_output, dropout)
        
        self.distmult = nn.Parameter(torch.rand(no_output))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        self.init =  torch.rand(self.K, no_output)
        
    def forward(self, x, adj, num_iter=1):
        embeds = self.GCN(x, adj)
        mu_init, _, dist = cluster(embeds, self.K, num_iter, cluster_temp = self.cluster_temp, init = self.init)
        mu, r, dist_2 = cluster(embeds, self.K, 1, cluster_temp = self.cluster_temp, init = mu_init.detach().clone())
        
        return r, dist
    
class GCNDeep(nn.Module):
    '''
    A stack of 4-layers GCNs.
    Input layer: no_features -> no_hidden
    Middle layer1: no_hidden->no_hidden
    Middle layer2: no_hidden->no_hidden
    Output layer: no_hidden->no_outpuy
    '''
    def __init__(self, no_features, no_hidden, no_output, dropout):
        super(GCNDeep, self).__init__()

        self.gc_input_layer = GraphConvolution(no_features, no_hidden)
        self.gcn_middle_layer_1 = GraphConvolution(no_hidden, no_hidden)
        self.gcn_middle_layer_2 = GraphConvolution(no_hidden, no_hidden)
        self.gc_output_layer = GraphConvolution(no_hidden, no_output)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc_input_layer(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gcn_middle_layer_1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gcn_middle_layer_2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc_output_layer(x, adj)
        return x
    
    
class GCNLink(nn.Module):
    '''
    GCN link prediction model 
    '''
    def __init__(self, no_features, no_hidden, no_output, dropout):
        super(GCNLink, self).__init__()

        self.GCN = GCN(no_features, no_hidden, no_output, dropout)
        self.distmult = nn.Parameter(torch.rand(no_output))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, adj, to_pred):
        embeds = self.GCN(x, adj)
        dot = (embeds[to_pred[:, 0]]*self.distmult.expand(to_pred.shape[0], self.distmult.shape[0])*embeds[to_pred[:, 1]]).sum(dim=1)
        return dot