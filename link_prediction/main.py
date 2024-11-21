import torch
from link_prediction.cgcnn import device, model, graph, node_transformer
from similarity import calculate_similarity
from characterNetwork_combined import load_edgelist_to_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 테스트용 그래프 (예제)
test_graph = load_edgelist_to_graph('./graphs/test_graph.edgelist').to(device)

# 모델로 노드 특성 추출
output_vectors = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr)
scalar_outputs = node_transformer(output_vectors).view(-1)

# 노드 간 유사도 계산
similarity_matrix = calculate_similarity(scalar_outputs)

# 유사도가 가장 높은 노드 쌍 예측
i, j = torch.triu_indices(similarity_matrix.size(0), similarity_matrix.size(1), 1)
predicted_indices = similarity_matrix[i, j].argmax()
predicted_nodes = (i[predicted_indices], j[predicted_indices])

print(f"Predicted Node Pair: {predicted_nodes}")