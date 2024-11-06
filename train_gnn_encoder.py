# train_encoder.py

import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from model_gnn_encoder import GNNEncoder, GNNClassifier
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def load_data(data_path):
    train_dataset = torch.load(os.path.join(data_path, 'train.pt'))
    val_dataset = torch.load(os.path.join(data_path, 'val.pt'))
    test_dataset = torch.load(os.path.join(data_path, 'test.pt'))
    return train_dataset, val_dataset, test_dataset


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            probs = F.softmax(out, dim=1)

            correct += (pred == data.y).sum().item()
            total += data.num_graphs

            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())
            all_probs.append(probs.cpu())

    accuracy = correct / total

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    num_classes = model.num_classes
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels.numpy(), all_probs[:, 1].numpy())
        else:
            auc = roc_auc_score(all_labels.numpy(), all_probs.numpy(), multi_class='ovr')
    except ValueError:
        auc = float('nan')
        print("AUC calculation error, possibly due to single class in samples.")

    return accuracy, auc


def main():
    parser = argparse.ArgumentParser(description='Train GNN Encoder')
    parser.add_argument('--dataset', type=str, default="BZR", 
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./splited_data', 
                        help='Root directory for split data')
    parser.add_argument('--save_path', type=str, default='./models_gnn_encoder', 
                        help='Path to save models')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                        help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    set_random_seed(args.seed)

    device = args.device

    data_path = os.path.join(args.data_root, args.dataset)
    train_dataset, val_dataset, test_dataset = load_data(data_path)

    sample_data = train_dataset[0]
    input_dim = sample_data.num_node_features
    num_classes = len(set([data.y.item() for data in train_dataset]))

    encoder = GNNEncoder(input_dim=input_dim, hidden_dim=args.hidden_dim)
    model = GNNClassifier(encoder=encoder, hidden_dim=args.hidden_dim, num_classes=num_classes)
    model.num_classes = num_classes
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    best_val_auc = 0.0

    epoch_iter = tqdm(range(1, args.epochs + 1), desc='Training', unit='epoch')

    for epoch in epoch_iter:
        loss = train(model, train_loader, optimizer, device)
        train_acc, train_auc = evaluate(model, train_loader, device)
        val_acc, val_auc = evaluate(model, val_loader, device)

        epoch_iter.set_postfix({
            'Loss': f'{loss:.4f}',
            'Train Acc': f'{train_acc:.4f}',
            'Train Auc': f'{train_auc:.4f}',
            'Val Acc': f'{val_acc:.4f}',
            'Val Auc': f'{val_auc:.4f}'
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_encoder_state = model.encoder.state_dict()

    test_acc, test_auc = evaluate(model, test_loader, device)
    print(f'Test Acc: {test_acc:.4f}, Test Auc: {test_auc:.4f}')

    os.makedirs(args.save_path, exist_ok=True)
    encoder_save_path = os.path.join(args.save_path, f'{args.dataset}_encoder.pth')
    torch.save(best_encoder_state, encoder_save_path)
    print(f'Encoder saved to {encoder_save_path}')


if __name__ == '__main__':
    main()
