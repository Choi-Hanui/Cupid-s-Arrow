import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.5):
        """
        Initialize the MLP model.
        
        Args:
        - input_size (int): Number of input features.
        - hidden_sizes (list[int]): List of sizes for the hidden layers.
        - output_size (int): Number of output features (e.g., classes for classification).
        - dropout_prob (float): Probability for dropout layers. Default is 0.5.
        """
        super(MLP, self).__init__()
        
        # Define the layers
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))  # Linear layer
            layers.append(nn.ReLU())  # Activation
            layers.append(nn.Dropout(p=dropout_prob))  # Dropout for regularization
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, output_size))  # Output layer
        
        # Combine layers in a Sequential container
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the MLP.
        
        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        
        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.model(x)



def calculate_similarity(scalar_outputs,label):
    num_nodes = scalar_outputs.size(0)
    similarity_matrix = []

    character1=label[0].item()
    character2=label[1].item()
    row=[]
    
    for i in range(num_nodes):
        if i==character1:
            row.append(torch.tensor(-np.inf).to(device))
        else:
            row.append(torch.mm(F.normalize(scalar_outputs[i],dim=0).unsqueeze(0), F.normalize(scalar_outputs[character1],dim=0).unsqueeze(1))[0][0])
    similarity_matrix.append(torch.stack(row))
    row=[]
    for i in range(num_nodes):
        if i==character2:
            row.append(torch.tensor(-np.inf).to(device))
        else:
            row.append(torch.mm(F.normalize(scalar_outputs[i],dim=0).unsqueeze(0), F.normalize(scalar_outputs[character2],dim=0).unsqueeze(1))[0][0])
    similarity_matrix.append(torch.stack(row))

    similarity_matrix = torch.stack(similarity_matrix).to('cuda')  # Combine rows into a matrix

    # Flatten, apply softmax, and reshape back
    similarity_matrix_stable = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0]  # 최대값 제거
    similarity_matrix_softmax = F.softmax(similarity_matrix_stable/1.5, dim=1)
    similarity_matrix = similarity_matrix_softmax  # Reshape back to original shape
    
    return similarity_matrix