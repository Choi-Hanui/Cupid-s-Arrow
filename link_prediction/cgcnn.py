import torch
import torch.nn.functional as F
from torch_geometric.nn import CGConv
from torch_geometric.data import DataLoader, Data
import networkx as nx
from similarity import calculate_similarity
from linear_trans import NodeFeatureToScalar

# 1. CGCNN 모델 정의
class CGCNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels):
        super(CGCNN, self).__init__()
        self.conv1 = CGConv(node_features, edge_features)
        self.conv2 = CGConv(hidden_channels, edge_features)
        self.fc = torch.nn.Linear(hidden_channels, 1)  # 노드 특성을 변환하는 선형 레이어

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return self.fc(x)

# 2. 그래프 데이터 준비
def load_edgelist_to_graph(file_path):
    '''
    Load an edgelist file and convert it to PyTorch Geometric Data.
    :param file_path: Path to the .edgelist file.
    :return: A Data object for PyTorch Geometric.
    '''
    G = nx.read_edgelist(file_path, data=[('co_occurrence', float), ('sentiment', float)])
    
    edge_index = torch.tensor(list(G.edges)).T.long()
    edge_attr = []
    for u, v in G.edges:
        edge_attr.append([G[u][v]['co_occurrence'], G[u][v]['sentiment']])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    x = torch.rand((len(G.nodes), 3))
    y = torch.zeros(len(G.nodes), dtype=torch.float)

    # Node name mapping
    node_mapping = {name: idx for idx, name in enumerate(G.nodes)}
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y), node_mapping

# 소설 이름 리스트와 라벨 데이터 준비
novelist = ["PrideAndPrejudice", "AnotherNovel"]
graphs = []
raw_labels = {
    "PrideAndPrejudice": ("Elizabeth", "Darcy"),  # 단일 라벨
    "AnotherNovel": ("Harry", "Hermione")        # 단일 라벨
}

for name in novelist:
    graph, mapping = load_edgelist_to_graph(f'./graphs/graph_{name}.edgelist')
    label = torch.tensor([mapping[raw_labels[name][0]], mapping[raw_labels[name][1]]])
    graphs.append((graph, label))

# DataLoader 준비
loader = DataLoader([g for g, _ in graphs], batch_size=1)

# 모델 및 학습 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CGCNN(node_features=3, edge_features=2, hidden_channels=16).to(device)
node_transformer = NodeFeatureToScalar(input_dim=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습 루프
for epoch in range(100):
    model.train()
    total_loss = 0

    for (graph, label) in graphs:
        graph = graph.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        # 노드 특성 추출
        output_vectors = model(graph.x, graph.edge_index, graph.edge_attr)

        # 노드 특성을 스칼라로 변환
        scalar_outputs = node_transformer(output_vectors).view(-1)

        # 유사도 계산
        similarity_matrix = calculate_similarity(scalar_outputs)

        # 예측된 노드 쌍 (유사도가 가장 높은 노드 2개)
        i, j = torch.triu_indices(similarity_matrix.size(0), similarity_matrix.size(1), 1)
        predicted_indices = similarity_matrix[i, j].argmax()
        predicted_nodes = torch.tensor([i[predicted_indices], j[predicted_indices]], dtype=torch.long)

        # 손실 계산
        loss = F.mse_loss(predicted_nodes.float(), label.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
