from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
import heapq

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from log import get_logger
import datetime
from torch_geometric.utils.convert import to_networkx
from sklearn.manifold import TSNE
from CalModularity import Q
import torch_geometric
import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='ClusterNet Training')


    parser.add_argument('--epochs', default=1001, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    # parser.add_argument('--label', default = 0, type = int, help = 'label pattern for node class')
    parser.add_argument('--num_class', default = 5, type = int, help = 'number of class for clustering')
    parser.add_argument('--num_edge', default = 3, type = int, help = 'number of edge: num_province*num_edge')
    
    parser.add_argument('--wd', default = 5e-4, type = float, help = 'weight decay')

    # parser.add_argument('--output', default = '/home/zhangziyi/code/ProvinceCuisineDataMining/Config')
    
    
    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args

args = get_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# from torch.utils.tensorboard import SummaryWriter


edge_filepath = 'YOUR PATH'
node_filepath = 'YOUR PATH'
label_filepath = 'YOUR PATH'


K = args.num_class

#initialize
edge_index = []
start = []
end = []

edge_attr = []
node_feature = []
label = []
province_list = []
label_pattern = 1

def isTopK(data, data_list, k=args.num_edge):
    
    return data >= sorted(data_list, reverse=True)[k-1]

def excel2edge(edge_filepath, k=args.num_edge): 
    df = pd.read_excel(edge_filepath, engine='openpyxl', sheet_name=0) 
    rows = df.shape[0]
    columns = df.shape[1]

    edge_weights = {}
    
    
    for i in range(columns):
        for j in range(rows):
            if i != j:
                weight = df.iloc[j, i] + df.iloc[i, j]
                if (i, j) in edge_weights:
                    edge_weights[(i, j)] += weight
                else:
                    edge_weights[(i, j)] = weight

    
    edge_list = []
    for i in range(columns):
        
        edges = [(j, edge_weights[(i, j)]) for j in range(columns) if (i, j) in edge_weights]
        edges = sorted(edges, key=lambda x: x[1], reverse=True)[:k]
        for j, weight in edges:
            edge_list.append((i, j, weight))
    
    
    start = []
    end = []
    edge_attr = []
    for edge in edge_list:
        start.append(edge[0])
        end.append(edge[1])
        edge_attr.append(edge[2])

    edge_index = [start, end]
    return edge_index, edge_attr

def excel2node():
    df = pd.read_excel(f'{node_filepath}', engine='openpyxl', sheet_name=0)  

    # print(df.shape)
    rows = df.shape[0]
    columns = df.shape[1]

    # print(rows)
    # print(columns)

    df_new = df.drop(df.columns[[0,1]], axis=1)
    # print(df_new.shape)
    global node_feature
    node_feature = torch.tensor(df_new.values, dtype=torch.float)

    # print(node_feature)
    # print(node_feature.size())

    return node_feature

 


def addClass():
    df = pd.read_excel(f'{label_filepath}', engine='openpyxl', sheet_name=label_pattern)
    
    global label
    
    label = torch.tensor(df['class'], dtype=torch.int64)
    print(label)
     
    
    # print(label)
    # print(label.size())
    return label


def getProvince():
    df = pd.read_excel(f'{label_filepath}', engine='openpyxl', sheet_name=0)
    global province_list
    province_list = df.values[: , 0]
    return province_list

def getMask():
    mask = []
    for i in range(31):
        mask.append(True)
    mask = torch.tensor(mask)
        
    
    # print(df_new['max_idx'])
    
    
    
     
    
    # print(label)
    # print(label.size())
    return mask
# build the PyG Dataset






 
#
class MyOwnDataset(InMemoryDataset):
    def __init__(self, 
                 root, 
                 transform=None, 
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def process(self):
        # Read data into huge `Data` list.
        node_feature=excel2node()
        edge_index, edge_attr=excel2edge()
        label = addClass()

        data = Data(x=node_feature, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr, 
                    y=addClass(),
                    train_mask=getMask())

        data_list = [data]
        
 
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
 
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
 
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = MyOwnDataset(root='YOUR PATH'+str(label_pattern)+'/')

# print(dataset.num_classes) # 0
# print(dataset[0].num_nodes) # 31
# print(dataset[0].num_edges) # 93
# print(dataset[0].num_features) # 8

data_list = [dataset[0]]
data = dataset[0]
# print(data)
# print(type(data))

G = to_networkx(data)



def calEuclidean(data, center):
    dist = np.sqrt(np.sum(np.square(data-center))) 
    # print(type(dist))
    return dist

def dict2list(r):
    com = []
    for i in range(K):
        l = r[i]
        com.append(l)
    return com    

def conductance(G, communities):
    cond = []
    for community in communities:
        subgraph = G.subgraph(community)
        internal_edges = subgraph.number_of_edges()
        boundary_edges = sum(1 for node in subgraph if any(neighbor not in community for neighbor in G.neighbors(node)))
        cond.append(boundary_edges / (2 * internal_edges + boundary_edges))
    return sum(cond) / len(cond)

def performance(G, communities):
    intra_edges = 0
    inter_edges = 0
    for community in communities:
        for node in community:
            for neighbor in G.neighbors(node):
                if neighbor in community:
                    intra_edges += 1
                else:
                    inter_edges += 1
    intra_edges /= 2  
    total_edges = G.number_of_edges()
    performance = (intra_edges + (total_edges - inter_edges)) / total_edges
    return performance

def calculate_coverage(G, communities):
    total_edges = G.number_of_edges()
    intra_edges = 0
    
    for community in communities:
        subgraph = G.subgraph(community)
        intra_edges += subgraph.number_of_edges()
    
    coverage = intra_edges / total_edges
    return coverage




def k_means(m, K):
    
    # print(m)
    nodes = list(G.nodes)
    
    centers = []
    temp = []
    for i in range(K):
        t = np.random.randint(0, len(nodes) - 1)
        # print(nodes[t])
        # print(m[nodes[t]])
        # print(centers)
        # if m[nodes[t]] not in centers:
        if nodes[t] not in temp:
            temp.append(nodes[t])
            centers.append(m[nodes[t]])  

    
    res = {}
    for i in range(K):
        res[i] = []

    for time in range(50):
        # clear
        for i in range(K):
            res[i].clear()
       
        nodes_distance = {}
        for node in nodes:
            
            node_distance = []
            for center in centers:
                # print('m[node]:',node, type(m[node]))
                # print('center:', center, type(center))
                node_distance.append(calEuclidean(m[node], center))  
            nodes_distance[node] = node_distance  
        
        # print(i)
        for node in nodes:
            temp = nodes_distance[node]  

            # print(temp)
            # print(min(temp))
            cls = temp.index(min(temp))
            res[cls].append(node)

       
        centers.clear()
        for i in range(K):
            center = []
            for j in range(dataset.num_classes):
                t = [m[node][j] for node in res[i]]  
                center.append(np.mean(t))
            centers.append(center)
           

    return res
# transform = RandomLinkSplit(is_undirected=True)
# print(type(data))
# data = transform(data)
# data = train_test_split_edges(data)
# print(data)
# print(type(data))
# print(data.x)
# loader = DataLoader(data_list, batch_size=1)
def draw_nx(com):
    
    pos = nx.spring_layout(G) 
    NodeId    = list(G.nodes())
    # print(f'NodeId:{NodeId}')
    # logger.info(f'NodeId:{NodeId}')
    node_size = [G.degree(i)**1.2*90 for i in NodeId] 

    plt.figure(figsize = (8,8)) 
    # print(pos)
    # print(type(com[1]))
    nx.draw(G,pos, 
            with_labels=True, 
            node_size =node_size, 
            node_color='w', 
            node_shape = '.'
        )

    
    color_list = [
            'pink','orange','r','g','b','y','m','gray','c','brown', '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
            '#ffd700']
    # print(len(com))
    # print(f'type of com:{type(com)}')
    # print(f'type of pos:{type(pos)}')
    for i in range(K):
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=com[i], 
                            node_color = color_list[i+2],  
                            label=True)
    plt.show()
    plt.savefig('YOUR PATH'+start_time[:10]+'/'+start_time[11:]+'/GCN-kmeans_nx')



def draw(z,r):
    colors = [
        'pink','orange','r','g','b','y','m','gray','c','brown', '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700']
    
    
    z = TSNE(n_components=2).fit_transform(z)
    province_list = getProvince()
    # z = np.c_[z,province_list]
    
    # z.append(province_list)
    # print(z)
    
    result = r
    # print(type(result))
    plt.figure(figsize=(8, 4))
    
    # print(result.get(1))
    # print(z[result.get(1),0])
    for j in range(K):
        plt.scatter(z[result.get(j),0], z[result.get(j),1],s=400, color=colors[j], alpha=0.5)
    plt.savefig('YOUR PATH'+start_time[:10]+'/'+start_time[11:]+'/GCN-kmeans.pdf', format='pdf')  

    for i in range(z.shape[0]):  ## for every node
        plt.annotate(province_list[i], xy = (z[i,0],z[i,1]),  xytext=(-20, 10), textcoords = 'offset points',ha = 'center', va = 'top',fontsize=10)    
    
    
    
        

    plt.show()
    plt.savefig('YOUR PATH'+start_time[:10]+'/'+start_time[11:]+'/GCN-kmeans1.pdf', format='pdf')  

start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger = get_logger(start_time)
# logger.get_logger()
# logger.add_handler(start_time)
logger.info("Begin")
logger.info(f'label:{label_pattern}')
logger.info(f'K：{K}')

hidden_dim = 16


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, dataset.num_classes)
    
    
    def forward(self):
        x, edge_index, edge_weight = data.x, torch.tensor(data.edge_index, dtype=torch.int), torch.tensor(data.edge_attr, dtype=torch.float)  #赋值data.x特征向量edge_index图的形状，edge_attr权重矩阵


        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)   
        
        x = F.dropout(x, training=self.training)   
        x = self.conv2(x, edge_index, edge_weight)  
        # print(x)
        x = F.log_softmax(x, dim=-1) 
        # print(x)
        return x   #
       

 
    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = data.to(device)
model = Net().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
logger.info(f'lr = {args.lr}  weight_decay:{args.wd}')


iter_num = 1001  
for epoch in range(iter_num):
    optimizer.zero_grad()
    model.train()
    out = model()
    # print('out:',out)
    label = data.y
    # one_hot = F.one_hot(label, num_classes = dataset.num_classes)
    # print(one_hot)
    
    loss = F.nll_loss(model()[data.train_mask], label[data.train_mask])
    loss.requires_grad_(True)
    loss.backward()
     
    optimizer.step()
     
    logger.info(f'epoch{epoch + 1}   loss:{loss}')
    print(f'epoch{epoch + 1}   loss:{loss}')
    if epoch == iter_num-1:
        out = out.detach().cpu().numpy()
        result = k_means(out, K)
        logger.info(f'Final Result:{result}')
        print(f'Final Result:{result}')
        print(type(result))
        # print(data.edge_index.cpu())
        # adj_array = edge_index_to_adj(data.edge_index.cpu())

        adj_array = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index)
        adj_array = adj_array.toarray()
        # print(adj_array)
        #计算模块度
       
        draw(out,result)
        draw_nx(result)
        com = dict2list(result)
        print(f'Modularity：{nx.community.modularity(G, com)}')
        logger.info(f'Modularity：{nx.community.modularity(G, com)}')

        # cond = conductance(G, com)
        # print(f"Conductance: {cond}")
        # logger.info(f"Conductance: {cond}")

        coverage = calculate_coverage(G, com)
        print(f"Coverage: {coverage}")
        logger.info(f"Coverage: {coverage}")

        perf = performance(G, com)
        print(f"Performance: {perf}")
        logger.info(f"Performance: {perf}")
        # print(data.y)
    





# if __name__=='__main__':
#     K_means()

