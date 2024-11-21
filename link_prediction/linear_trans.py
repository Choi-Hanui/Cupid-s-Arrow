import torch

class NodeFeatureToScalar(torch.nn.Module):
    def __init__(self, input_dim):
        super(NodeFeatureToScalar, self).__init__()
        self.fc = torch.nn.Linear(input_dim, 1)  # 입력 차원을 스칼라로 매핑

    def forward(self, x):
        return self.fc(x)