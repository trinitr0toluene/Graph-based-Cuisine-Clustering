import numpy as np
import scipy as sp
import scipy.sparse
import torch

def negative_sample(edges, num_samples, bin_adj_train):
    all_edges = torch.zeros(edges.shape[0]*(num_samples+1), 2).long()
    all_edges[:edges.shape[0]] = edges
    labels = torch.zeros(all_edges.shape[0])
    labels[:edges.shape[0]] = 1
    idx = edges.shape[0]
    n = bin_adj_train.shape[0]
    for i in range(edges.shape[0]):
        for j in range(num_samples):
            #draw negative samples by randomly changing either the source or the destination node of this edge
            if np.random.rand() < 0.5: 
                to_replace = 0
            else: 
                to_replace = 1
            all_edges[idx, 1-to_replace] = edges[i, 1-to_replace]
            all_edges[idx, to_replace] = np.random.randint(0, n-1)
            while bin_adj_train[all_edges[idx, 0], all_edges[idx, 1]] == 1:
                all_edges[idx, to_replace] = np.random.randint(0, n-1)
            idx += 1
    return all_edges, labels

def edge_dropout(edges, p_remove):
    indices = np.arange(edges.shape[0])
    keep = np.random.choice(indices, int((1 - p_remove)*edges.shape[0]), replace=False)
    return edges[keep]

def make_normalized_adj(edges, n):
    edges = edges.detach().numpy()
    adj = sp.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.sparse.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj)

def load_data(path, dataset):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.sparse.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    values = np.unique(idx_features_labels[:, -1])
    values.sort()
    labels = np.zeros(idx_features_labels.shape[0])
    for i in range(labels.shape[0]):
        labels[i] = np.where(values == idx_features_labels[i, -1])[0][0]
    labels = torch.tensor(labels).long()
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.sparse.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    adj = adj.coalesce()
    binary_adj = (adj.to_dense()>0).float()
    
    return adj, binary_adj, features


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
