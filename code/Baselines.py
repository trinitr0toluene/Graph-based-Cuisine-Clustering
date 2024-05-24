import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities as lpa
from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import heapq
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset
from log import get_logger
import datetime
from torch_geometric.utils.convert import to_networkx
import torch_geometric
from communities.algorithms import louvain_method
from communities.visualization import draw_communities
from communities.visualization import louvain_animation
from CalModularity import Q
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
k = args.num_edge

#initialize
edge_index = []
start = []
end = []

edge_attr = []
node_feature = []
label = []
province_list = []
label_pattern = 0

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
    df_new = df.drop(df.columns[[0,1]], axis=1)
    df_new['max_idx'] = df_new.idxmax(axis=1)
    # print(df_new['max_idx'])
    global label
    
    label = torch.tensor(df_new['max_idx'], dtype=torch.int64)
     
    
    # print(label)
    # print(label.size())
    return label


def getProvince():
    df = pd.read_excel(f'{label_filepath}', engine='openpyxl', sheet_name=0)
    global province_list
    province_list = df.values[: , 0]
    return province_list

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




def draw(z,r,filename):
    colors = [
            'pink','orange','r','g','b','y','m','gray','c','brown', '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
            '#ffd700']
    
    province_list = getProvince()

    K = len(r)
    # z = z.values()
    z = np.array(list(z.values()))
    # print(z)
    result = dict()
    for index, item in enumerate(r):
        result[index] = list(item)
    # print(result)
    # result = dict(enumerate(com))
    # print(f'type of result:{type(result)}'  f'result:{result}')
    # print(f'type of z:{type(z)}'  f'result:{z}')
    plt.figure(figsize=(8, 8))
    
    # print(result.get(1))
    # print(z[result.get(1),0])
    for j in range(K):
        plt.scatter(z[result.get(j),0], z[result.get(j),1],s=450, color=colors[j], alpha=0.5)
    for i in range(z.shape[0]):  ## for every node
        plt.annotate(province_list[i], xy = (z[i,0],z[i,1]),  xytext=(-20, 10), textcoords = 'offset points',ha = 'center', va = 'top')    

    plt.axis('off')
    plt.show()
    plt.savefig('YOUR PATH'+start_time[:10]+'/'+start_time[11:]+'/'+filename+'_scatter')
  



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
   
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    
    def process(self):
        # Read data into huge `Data` list.
        node_feature=excel2node()
        edge_index, edge_attr=excel2edge()
        label = addClass()

        data = Data(x=node_feature, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr, 
                    y=addClass(),
                    )

        data_list = [data]
        
 
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
 
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
 
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = MyOwnDataset(root='YOUR PATH'+str(label_pattern)+'/')

data_list = [dataset[0]]
data = dataset[0]


start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger = get_logger(start_time)
logger.info("Begin")
logger.info(f'label:{label_pattern}')

def draw_nx(com,filename):
    
    
    NodeId    = list(G.nodes())
    # print(f'NodeId:{NodeId}')
    # logger.info(f'NodeId:{NodeId}')
    node_size = [G.degree(i)**1.2*90 for i in NodeId] # 节点大小

    plt.figure(figsize = (8,8)) # 设置图片大小
    # print(pos)
    # print(type(com[1]))
    nx.draw(G,pos, 
            with_labels=True, 
            node_size =node_size, 
            node_color='w', 
            node_shape = '.'
        )

   
    color_list = ['pink','orange','r','g','b','y','m','gray','c','brown', '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
                '#ffd700']
    # print(len(com))
    # print(f'type of com:{type(com)}')
    # print(f'type of pos:{type(pos)}')
    for i in range(len(com)):
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=com[i], 
                            node_color = color_list[i+2],  
                            label=True)
    plt.show()
    plt.savefig('YOUR PATH'+start_time[:10]+'/'+start_time[11:]+'/'+filename+'_nx')

G = to_networkx(data)
pos = nx.spring_layout(G) # 节点的布局为spring型

def lpa_only():
    adj_matrix = nx.to_numpy_array(G)
    com = list(lpa(G))
    print('LPA')
    print(f'community nums:{len(com)}'  f'result：{com}')
    logger.info(f'LPA' f'community nums:{len(com)}'  f'result：{com}')

    draw(pos, com,filename = 'LPA')
    draw_nx(com, filename = 'LPA')
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
    
    # # draw_communities(adj_matrix, com, filename = '/home/zhangziyi/code/ProvinceCuisineDataMining/Log/'+start_time[:10]+'/'+start_time[11:]+'/LPA_plot')
    # calculate(com)

def louvain_only():
    adj_matrix = nx.to_numpy_array(G)
    com, frames = louvain_method(adj_matrix, n=None)
    print('Louvain')
    print(f'community nums:{len(com)}'  f'result：{com}')
    logger.info(f'Louvain:' f'community nums:{len(com)}'  f'result：{com}')

    draw(pos, com, filename = 'Louvain')
    draw_nx(com, filename = 'Louvain')
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

def kmeans_only():
    adj_matrix = nx.to_numpy_array(G)
    kmeans = KMeans(n_clusters=args.num_class, random_state=0).fit(data.x)
    labels = kmeans.labels_

    
    node_colors = [labels[node] for node in G.nodes()]
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))

  
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, cmap=plt.cm.viridis)
    
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    nx.draw_networkx_labels(G, pos)

    plt.title('Community Detection using K-means')
    plt.show()
    plt.savefig('YOUR PATH'+start_time[:10]+'/'+start_time[11:]+'/'+'kmeans-only'+'_nx')
    communities = {node: int(label) for node, label in zip(G.nodes(), labels)}

    community_groups = {}
    for node, community in communities.items():
        if community not in community_groups:
            community_groups[community] = []
        community_groups[community].append(node)

    com = list(community_groups.values())
    
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


if __name__== "__main__" :
    # lpa_only()
    # louvain_only()
    kmeans_only()


