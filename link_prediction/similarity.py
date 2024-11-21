import torch

def calculate_similarity(scalar_outputs):
    num_nodes = scalar_outputs.size(0)
    similarity_matrix = torch.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            similarity_matrix[i, j] = torch.dot(scalar_outputs[i], scalar_outputs[j])
    return similarity_matrix