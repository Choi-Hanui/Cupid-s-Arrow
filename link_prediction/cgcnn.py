# python -m link_prediction.cgcnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv
from torch_geometric.data import DataLoader, Data
import networkx as nx
from link_prediction.similarity import calculate_similarity
from link_prediction.linear_trans import NodeFeatureToScalar
import numpy as np
import pandas as pd
import re
import difflib

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        # Define the layers
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))  # Linear layer
            layers.append(nn.ReLU())  # Activation
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, output_size))  # Output layer
        
        # Combine layers in a Sequential container
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):

        return self.model(x)

class CGCNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels):
        super(CGCNN, self).__init__()
    
        # 첫 번째 CGConv 계층: node_features -> hidden_channels
        self.embedding_node=nn.Linear(node_features,hidden_channels)
        self.embedding_edge=nn.Linear(edge_features,hidden_channels)
        
        self.conv1 = CGConv(channels=(hidden_channels, hidden_channels), dim=hidden_channels)
        # 두 번째 CGConv 계층: hidden_channels -> hidden_channels
        self.conv2 = CGConv(channels=(hidden_channels, hidden_channels), dim=hidden_channels)
        # 최종 선형 계층: hidden_channels -> 1
        self.attention_weight=nn.Linear(hidden_channels,hidden_channels)
        
    def forward(self, x, edge_index, edge_attr):
        
        # 첫 번째 CGConv
        x=self.embedding_node(x)
        edge_attr=self.embedding_edge(edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        # 두 번째 CGConv
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x=self.attention_weight(x)
        return x
    




def parse_edge_data(file_path):
    """
    Parse edge data from a file where each line is in the format:
    node1 node2 { 'key1': value1, 'key2': value2 }
    """
    G = nx.Graph()

    # Regular expression to parse edge attributes
    pattern = re.compile(r"(\w+)\s+(\w+)\s+\{(.*?)\}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                node1, node2, attributes = match.groups()
                
                # Manually parse the attributes
                attributes = attributes.replace("np.float64", "").replace("(", "").replace(")", "")
                attr_dict = dict(
                    (key.strip("'\""), float(value))
                    for key, value in (item.split(":") for item in attributes.split(","))
                )

                # Ensure all required attributes are present
                attr_dict.setdefault('co_occurrence', 0.0)  # Default value if missing
                attr_dict.setdefault('sentiment', 0.0)     # Default value if missing

                # Add edge with attributes to the graph
                G.add_edge(node1, node2, **attr_dict)

    return G


def find_closest_name(target_name, node_names):
    """
    Find the closest match to `target_name` in `node_names` using difflib.
    """
    matches = difflib.get_close_matches(target_name.lower(), node_names, n=1, cutoff=0.6)
    return matches[0] if matches else None

def load_graph_from_lists(edgelist_path, nodelist=None):
    """
    Load graph data from edgelist and optional nodelist.
    """
    # Parse edgelist using `parse_edge_data`
    G = parse_edge_data(edgelist_path)
    node_gender=[]
    # Add nodes from nodelist if provided
    if nodelist:
        if isinstance(nodelist, str):
            with open(nodelist, 'r', encoding='utf-8') as f:
                nodes = [line.strip() for line in f.readlines()]
        else:
            nodes = nodelist
        
        for node in nodes:
            if node.endswith('F'):
                node_gender.append([0,1])
            elif node.endswith('M'):
                node_gender.append([1,0])
            else:
                print('error')
            if node not in G.nodes:
                G.add_node(node)

    # Build node attributes and mappings
    node_mapping = {name: idx for idx, name in enumerate(G.nodes)}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges], dtype=torch.long).T
    
    # Handle missing attributes with default values
    edge_attr = torch.tensor(
        [[
            G[u][v].get('co_occurrence', 0.0),  # Default to 0.0 if missing
            G[u][v].get('sentiment', 0.0)      # Default to 0.0 if missing
        ] for u, v in G.edges],
        dtype=torch.float
    )

    # Node features (random for now) and labels
    x = torch.tensor(node_gender)  # Example node features????
    
    y = torch.zeros(len(G.nodes), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y), node_mapping

def train(train_epoch=1000):
    # Example Usage:
    novelist = ["AnnaOfTheIsland","AnneOfAvonlea","PrideAndPrejudice","RomeoAndJuliet","TheAgeOfInnocence","ThePhantomOfTheOpera",
                "ARoomWithAView","Emma","JaneEyre","Middlemarch","Persuasion","SenseAndSensibility","TheGreatGatsby"]
                # test: "APairOfBlueEyes"

    graphs = []
    raw_labels = {
        # "APairOfBlueEyes": ("swancourt", "knight"), # 엘프리다 스완코트 & 헨리 나이트 & 스티븐 스미스 
        # "APairOfBlueEyes": ("swancourt", "smith"), # 셋이 삼각관계인데 누구와도 이루어지지 못하고 죽습니다
        "ARoomWithAView": ("honeychurch", "emerson"), # 루시 허니처치 & 조지 에머슨
        "AnnaOfTheIsland": ("anne", "gilbert"),  # 앤 셜리 & 길버트 블라이드
        "AnneOfAvonlea": ("anne", "gilbert"), # 앤 셜리 & 길버트 블라이드
        "Emma": ("woodhouse", "knightley"), # 엠마 우드하우스 & 조지 나이트리
        "JaneEyre": ("eyre", "rochester"), # 제인 에어 & 에드워드 로체스터
        "Middlemarch": ("dorothea", "ladislaw"), # 또는 (“brooke“, “ladislaw”) 도로시아 브룩 & 윌 러디슬로우
        "Persuasion": ("elliot", "wentworth"), # 또는 (“anne”, “wentworth”) 앤 엘리엇 & 프레데릭 웬트워스
        "PrideAndPrejudice": ("bennet", "darcy"), # 엘리자베스 베넷 & 미스터 다아시
        "RomeoAndJuliet": ("romeo", "juliet"), # 로미오 & 줄리엣
        "SenseAndSensibility": ("elinor", "ferras"), # 엘리너 대시우드 & 에드워드 패러스 
        "TheAgeOfInnocence": ("archer", "olenska"), # 뉴랜드 아처 & 엘렌 올렌스카
        "TheGreatGatsby": ("gatsby", "buchanan"), # 제이 개츠비 & 데이지 뷰캐넌
        "ThePhantomOfTheOpera": ("daae", "raoul") # 크리스틴 다에 & 라울 비콩트
    }



    for name in novelist:
        graph, mapping = load_graph_from_lists(
            edgelist_path=f'./graphs/{name} combined graph.edgelist',
            nodelist=f'./graphs/{name} gender.nodelist'
        )
        
        # Convert graph node names to lowercase
        lower_mapping = {node.lower(): node for node in mapping.keys() if isinstance(node, str)}
        node_names_lower = list(lower_mapping.keys())

        # Find closest matches for raw_labels
        raw_label_0, raw_label_1 = raw_labels[name]
        closest_0 = find_closest_name(raw_label_0, node_names_lower)
        closest_1 = find_closest_name(raw_label_1, node_names_lower)

        # Replace with closest matches and log changes
        if closest_0 and closest_1:
            #print(f"In {name}, replacing '{raw_label_0}' with '{closest_0}' and '{raw_label_1}' with '{closest_1}'")
            label = torch.tensor([mapping[lower_mapping[closest_0]], mapping[lower_mapping[closest_1]]])
            graphs.append((graph, label))
        else:
            print(f"Warning: Could not find a match for one or both labels in {name}. Skipping this graph.")



    loader = DataLoader([g for g, _ in graphs], batch_size=1)

    node_features = 2  # graph.x.shape[1]
    edge_features = 2  # graph.edge_attr.shape[1]
    hidden_channels = 64
    save_step=100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CGCNN(node_features=node_features, edge_features=edge_features, hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mlp=MLP(32, [32,16,8], 1).to(device)
    
    history=np.zeros(train_epoch+10)
    for epoch in range(train_epoch):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for graph, label in graphs:
            graph = graph.to(device)
            label = label.to(device)
            
            # Debugging shapes
            
            output_vectors = model(graph.x/5, graph.edge_index, graph.edge_attr)
            
            # 각각 남 여 캐릭터의 선호도
            '''
            num_nodes = output_vectors.size(0)
            similarity_matrix = []
            
            character1=label[0].item()
            character2=label[0].item()
            row=[]
            for i in range(num_nodes):
                row.append(mlp(torch.cat((output_vectors[i], output_vectors[character1]), dim=0))[0])
            similarity_matrix.append(torch.stack(row))
            row=[]
            for i in range(num_nodes):
                row.append(mlp(torch.cat((output_vectors[i], output_vectors[character2]), dim=0))[0])
            similarity_matrix.append(torch.stack(row))

            similarity_matrix = torch.stack(similarity_matrix).to('cuda')  # Combine rows into a matrix

            # Flatten, apply softmax, and reshape back
            similarity_matrix_stable = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0]  # 최대값 제거
            similarity_matrix_softmax = F.softmax(similarity_matrix_stable, dim=1)
            similarity_matrix = similarity_matrix_softmax  # Reshape back to original shape
            '''
            similarity_matrix=calculate_similarity(output_vectors,label)
            #print(similarity_matrix)
            character_num=graph.x.shape[0]
            label1=torch.zeros((2,character_num)).to('cuda')
            label1[0,label[1].item()]=1
            label1[1,label[0].item()]=1
            criterion = nn.MSELoss()
            loss = criterion(label1, similarity_matrix)
            total_loss += loss
        total_loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss.item():.4f}")
        history[epoch]=total_loss.item()
        if epoch%save_step==0:
            model_save_path = "./trained_model"+str(epoch)+'step.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to model_save_path")
    history=pd.DataFrame(history)
    history.to_excel('history.xlsx')
    