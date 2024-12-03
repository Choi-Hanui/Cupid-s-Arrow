import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def calculate_similarity(scalar_outputs):
    num_nodes = scalar_outputs.size(0)
    similarity_matrix = []

    for i in range(num_nodes):
        row=[]
        for j in range(num_nodes):
            if i==j:
                row.append(torch.tensor(-np.inf).to(device))
            else:
                row.append(torch.mm(scalar_outputs[i].unsqueeze(0), scalar_outputs[j].unsqueeze(1))[0][0])
        similarity_matrix.append(torch.stack(row))
    similarity_matrix = torch.stack(similarity_matrix).to('cuda')  # Combine rows into a matrix
    # Flatten, apply softmax, and reshape back
    similarity_matrix_stable = similarity_matrix - similarity_matrix.max().item()  # 최대값 제거
    similarity_matrix_softmax = F.softmax(similarity_matrix_stable)
    similarity_matrix = similarity_matrix_softmax  # Reshape back to original shape
    return similarity_matrix