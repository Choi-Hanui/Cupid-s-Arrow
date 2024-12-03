# python -m link_prediction.main

# main.py
import torch
import difflib
import warnings

# UserWarning 무시
warnings.filterwarnings("ignore", category=UserWarning)

from link_prediction.cgcnn import CGCNN  # 모델 클래스와 디바이스 가져오기
from link_prediction.similarity import calculate_similarity  # 유사도 계산 함수
from link_prediction.cgcnn import load_graph_from_lists,find_closest_name,train  # 그래프 로드 함수
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_closest_couple(target_name, node_names, gender_map, target_gender):
    """
    Find the closest match to `target_name` in `node_names` using difflib,
    restricted to nodes of a specific gender.
    
    Parameters:
    - target_name: The name to find a match for.
    - node_names: List of available node names.
    - gender_map: A dictionary mapping node names to their genders.
    - target_gender: The gender ('M' or 'F') to filter node names.
    
    Returns:
    - Closest node name of the specified gender or None.
    """
    # Filter node names by target gender
    filtered_names = [name for name in node_names if gender_map.get(name, '') == target_gender]
    
    # Find closest match among filtered names
    matches = difflib.get_close_matches(target_name.lower(), filtered_names, n=1, cutoff=0.6)
    return matches[0] if matches else None


# 모델 파라미터 설정
node_features = 2  # 노드 특징 차원
edge_features = 2  # 엣지 특징 차원
hidden_channels = 64  # 은닉 채널 수

# 모델 초기화 및 로드
model = CGCNN(node_features=node_features, edge_features=edge_features, hidden_channels=hidden_channels).to(device)
iftrain=False
if iftrain:
    print('train_start')
    train(101)
model_load_path = "./trained_model100step.pth"
model.load_state_dict(torch.load(model_load_path, map_location=device,weights_only=True))
model.eval()  # 평가 모드로 전환
print(f"Model loaded from {model_load_path}")

# 테스트 데이터 로드
edge_list_path = './graphs/APairOfBlueEyes combined graph.edgelist'
node_list_path = './graphs/APairOfBlueEyes_gender.nodelist'
test_graph, mapping = load_graph_from_lists(edge_list_path, node_list_path)
test_graph = test_graph.to(device)


test_graph.x=test_graph.x.float()
test_graph.edge_attr.float()

# 모델로 노드 특성 추출
with torch.no_grad():
    output_vectors = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr)

# 노드 간 유사도 계산
similarity_matrix = calculate_similarity(output_vectors)



# 유사도가 가장 높은 노드 쌍 예측

max_value = torch.max(similarity_matrix)  # 최대값
max_index = torch.argmax(similarity_matrix)  # 1D로 펼쳐진 인덱스

# 2차원 텐서에서의 좌표 (row, col)
row, col = divmod(max_index.item(), similarity_matrix.size(1))

# 예측 결과 출력
reverse_mapping = {v: k for k, v in mapping.items()}  # 숫자 인덱스를 노드 이름으로 매핑
predicted_node_names = (reverse_mapping[row], reverse_mapping[col])
print(predicted_node_names[0]+' and '+predicted_node_names[1]+' have highest attention score')
print('May be '+predicted_node_names[0]+' fall in love to '+predicted_node_names[1])
