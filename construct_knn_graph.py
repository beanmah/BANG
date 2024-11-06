# construct_knn_graph.py

import torch
from grakel.kernels import ShortestPath
from torch_geometric.loader import DataLoader
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from model_gnn_encoder import GNNEncoder

def construct_knn(kernel_idx):
    """
    Construct edge indices (bidirectional edges) based on KNN indices.
    """
    num_graphs, knn_nei_num = kernel_idx.shape
    src = kernel_idx.reshape(-1)
    dst = torch.arange(num_graphs).unsqueeze(1).repeat(1, knn_nei_num).reshape(-1)
    edge_index = torch.cat((
        torch.stack([dst, src], dim=0),
        torch.stack([src, dst], dim=0)
    ), dim=1)
    return edge_index

def build_knn_graph_from_kernel(dataset, args):
    """
    Generate similarity matrix using Grakel's ShortestPath kernel and construct KNN graph.
    """
    grakel_graphs = []

    for data in dataset:
        edge_index = data.edge_index.T.tolist()
        edge_set = set((int(e[0]), int(e[1])) for e in edge_index)
        node_labels = {}
        edge_labels = dict()
        grakel_graphs.append([edge_set, node_labels, edge_labels])

    gk = ShortestPath(normalize=True, with_labels=False)
    kernel_matrix = gk.fit_transform(grakel_graphs)
    kernel_simi = torch.tensor(kernel_matrix, dtype=torch.float)

    kernel_idx = torch.topk(kernel_simi, k=args.knn_nei_num, dim=1, largest=True)[1][:, 1:]
    knn_edge_index = construct_knn(kernel_idx)

    return kernel_simi, kernel_idx, knn_edge_index

def build_knn_graph_from_embeddings(dataset, args):
    """
    Generate graph embeddings using pre-trained GNN model and construct KNN graph based on embedding similarity.
    """
    device = args.device
    input_dim = dataset[0].x.shape[1]
    hidden_dim = 128

    encoder = GNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    encoder.eval()

    model_path = os.path.join(args.pretrained_gnn_encoder_path, f'{args.dataset}_encoder.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model file not found: {model_path}")
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            batch_indices = batch.batch.to(device)
            embedding = encoder(x, edge_index, batch_indices)
            embeddings.append(embedding.cpu())

    embeddings = torch.cat(embeddings, dim=0)

    similarity_matrix = cosine_similarity(embeddings.numpy(), embeddings.numpy()) 
    similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float)

    kernel_idx = torch.topk(similarity_matrix, k=args.knn_nei_num, dim=1, largest=True)[1][:, 1:] 
    knn_edge_index = construct_knn(kernel_idx)

    return similarity_matrix, kernel_idx, knn_edge_index

def build_knn_graph_combined(dataset, args):
    """
    Combine kernel and embedding similarity matrices to generate a comprehensive KNN graph.
    """
    kernel_simi, _, _ = build_knn_graph_from_kernel(dataset, args)
    embedding_simi, _, _ = build_knn_graph_from_embeddings(dataset, args)

    scaler = MinMaxScaler()
    kernel_simi_np = kernel_simi.numpy()
    embedding_simi_np = embedding_simi.numpy()

    kernel_simi_normalized = scaler.fit_transform(kernel_simi_np)
    embedding_simi_normalized = scaler.fit_transform(embedding_simi_np)

    kernel_simi_normalized = torch.tensor(kernel_simi_normalized, dtype=torch.float)
    embedding_simi_normalized = torch.tensor(embedding_simi_normalized, dtype=torch.float)

    alpha = 0.5
    beta = 0.5
    combined_simi = alpha * kernel_simi_normalized + beta * embedding_simi_normalized

    combined_idx = torch.topk(combined_simi, k=args.knn_nei_num, dim=1, largest=True)[1][:, 1:]

    knn_edge_index_combined = construct_knn(combined_idx)

    return combined_simi, combined_idx, knn_edge_index_combined