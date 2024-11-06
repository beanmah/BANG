# infer_and_prepare_semi_supervised_data.py

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from parse import parse_args
from utils import set_seed
from model import GIN, GCN, MLP_Classifier
from dataset import Dataset
from construct_knn_graph import build_knn_graph_from_kernel, build_knn_graph_from_embeddings, build_knn_graph_combined


def predict(args, encoder, gnn, classifier, data_loader):
    encoder.eval()
    gnn.eval()
    classifier.eval()

    preds = []
    confidences = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            data = batch['data'].to(args.device)
            data_idx = batch['idx'].to(args.device)
            knn_edge_index = batch['knn_edge_index'].to(args.device)

            H = encoder(data.x, data.adj_t, data.ptr)
            H_knn = gnn(H, knn_edge_index)
            logits = classifier(H_knn)[data_idx]

            probabilities = torch.softmax(logits, dim=1)  # Assuming logits are raw scores
            confidence, pred = torch.max(probabilities, dim=1)

            preds.extend(pred.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())

    return preds, confidences

def save_processed_data(dataset, preds, confidences):
    # Create a new list to store processed graph data
    processed_data = []

    for data, pred, conf in zip(dataset, preds, confidences):
        # Add prediction labels and confidence to Data object
        data.predicted_label = torch.tensor([pred], dtype=torch.long)
        data.label_confidence = torch.tensor([conf], dtype=torch.float)
        processed_data.append(data)

    return processed_data

def main():
    args = parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)

    # Load data
    data_path = os.path.join(args.splited_data_path, args.dataset)
    train_data = torch.load(os.path.join(data_path, 'train.pt'))
    val_data = torch.load(os.path.join(data_path, 'val.pt'))
    test_data = torch.load(os.path.join(data_path, 'test.pt'))
    unlabel_train_data = torch.load(os.path.join(data_path, 'unlabel_train.pt'))


    # Set y and is_pseudo_label for original train_data, val_data, test_data, and unlabel_train_data
    for data in train_data:
        # True labels
        data.predicted_label = data.y  # data.y is the true label in the dataset
        data.label_confidence = torch.tensor([1.0], dtype=torch.float)  # Set default value as needed
        data.is_pseudo_label = torch.tensor([0], dtype=torch.long)  # 0 indicates using true label

    for data in val_data:
        # True labels
        data.predicted_label = data.y  # data.y is the true label in the dataset
        data.label_confidence = torch.tensor([1.0], dtype=torch.float)  # Set default value as needed
        data.is_pseudo_label = torch.tensor([0], dtype=torch.long)  # 0 indicates using true label

    for data in test_data:
        # True labels
        data.predicted_label = data.y  # data.y is the true label in the dataset
        data.label_confidence = torch.tensor([1.0], dtype=torch.float)  # Set default value as needed
        data.is_pseudo_label = torch.tensor([0], dtype=torch.long)  # 0 indicates using true label

    for data in unlabel_train_data:
        # Pseudo labels
        data.predicted_label = data.y  # Pseudo labels stored in data.predicted_label
        data.label_confidence = torch.tensor([1.0], dtype=torch.float)  # Set default value as needed
        data.is_pseudo_label = torch.tensor([1], dtype=torch.long)  # 1 indicates using pseudo label

    # Get feature and class counts
    all_data = train_data + val_data + test_data + unlabel_train_data  # Contains all data
    if len(all_data) > 0:
        args.n_feat = all_data[0].num_features
        args.n_class = len(set([data.y.item() for data in all_data if data.y is not None]))
    else:
        raise ValueError("No data available to determine n_feat and n_class.")

    total_samples = len(all_data)

    dataset_train_and_val = train_data + val_data + unlabel_train_data
    if args.knn_method == 'kernel':
        _, kernel_idx_train_and_val, knn_edge_index_train_and_val = build_knn_graph_from_kernel(dataset_train_and_val, args)
    elif args.knn_method == 'embedding':
        _, kernel_idx_train_and_val, knn_edge_index_train_and_val = build_knn_graph_from_embeddings(dataset_train_and_val, args)
    elif args.knn_method == 'combined':
        _, kernel_idx_train_and_val, knn_edge_index_train_and_val = build_knn_graph_combined(dataset_train_and_val, args)
    else:
        raise ValueError(f"Unknown KNN method: {args.knn_method}")
    
    unlabel_dataset = Dataset(unlabel_train_data, dataset_train_and_val, kernel_idx_train_and_val, knn_edge_index_train_and_val)

    # Create data loader
    unlabel_loader = DataLoader(unlabel_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=unlabel_dataset.collate_batch)

    # Initialize models
    encoder = GIN(args).to(args.device)
    gnn = GCN(args).to(args.device)
    classifier = MLP_Classifier(args).to(args.device)

    # Load model parameters
    model_save_path = os.path.join(args.model_path, f'model_{args.dataset}.pth')
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model file not found at {model_save_path}")

    checkpoint = torch.load(model_save_path, map_location=args.device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    gnn.load_state_dict(checkpoint['gnn_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    # Make predictions on unlabeled data
    preds, confidences = predict(args, encoder, gnn, classifier, unlabel_loader)

    # Save predictions to data
    processed_unlabel_data = save_processed_data(unlabel_train_data, preds, confidences)

    # Merge original train.pt and processed unlabel_train.pt
    combined_train_data = train_data + processed_unlabel_data

    # Create output directory
    output_dir = os.path.join(args.semi_supervised_data_path, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Save new train.pt
    torch.save(combined_train_data, os.path.join(output_dir, 'train.pt'))
    print(f"Combined train data saved to {os.path.join(output_dir, 'train.pt')}")

    # Save val.pt and test.pt to new directory
    torch.save(val_data, os.path.join(output_dir, 'val.pt'))
    torch.save(test_data, os.path.join(output_dir, 'test.pt'))
    print(f"Validation and test data saved to {output_dir}")

    # Calculate and print sample counts and percentages for each dataset
    combined_train_len = len(combined_train_data)
    val_len = len(val_data)
    test_len = len(test_data)

    print("\nDataset Split Statistics:")
    print(f"Training Set (train.pt): {combined_train_len} samples, {combined_train_len / total_samples * 100:.2f}%")
    print(f"Validation Set (val.pt): {val_len} samples, {val_len / total_samples * 100:.2f}%")
    print(f"Test Set (test.pt): {test_len} samples, {test_len / total_samples * 100:.2f}%")
    print(f"Total Samples: {total_samples}")

if __name__ == '__main__':
    main()