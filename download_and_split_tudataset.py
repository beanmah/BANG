
import os
import torch
import argparse
import random
import numpy as np
from dataset import TUDataset
from torch_geometric.transforms import ToSparseTensor, Constant
from utils import set_seed

def add_adj_t(data):
    adj_t = ToSparseTensor()(data).adj_t
    data.adj_t = adj_t
    return data

def process_labels(data):
    data.y = torch.where(data.y == 0, torch.tensor(1), torch.tensor(0))
    return data

def download_dataset(name, path):

    # If the dataset is ENZYMES, add label processing logic
    if name == 'ENZYMES':
        # dataset = TUDataset(root=path, name=name, pre_transform=add_adj_t)
        dataset = TUDataset(root=path, name=name, pre_transform=add_adj_t, transform=process_labels)
    # If the dataset is IMDB-BINARY or REDDIT-BINARY, set node features to constant 1
    elif name in ['IMDB-BINARY', 'REDDIT-BINARY']:
        transform = Constant(1, cat=False)
        # Download and apply the initial pre_transform (e.g., add adj_t) and append the node feature transform
        dataset = TUDataset(root=path, name=name, pre_transform=add_adj_t, transform=transform)
    else:
        dataset = TUDataset(root=path, name=name, pre_transform=add_adj_t)
            
    return dataset


def assign_ids(dataset):
    for idx, data in enumerate(dataset):
        data.id = torch.tensor([idx])
    return dataset

def compute_counts(n_cls, ratios):
    ratios = np.array(ratios)
    counts = np.floor(ratios * n_cls).astype(int)
    diff = n_cls - counts.sum()
    fractions = ratios * n_cls - counts
    # Adjust counts so that the sum equals n_cls
    while diff > 0:
        idx = np.argmax(fractions)
        counts[idx] += 1
        fractions[idx] = 0  # Prevent duplicate allocation
        diff -= 1
    while diff < 0:
        idx = np.argmin(fractions)
        counts[idx] -= 1
        fractions[idx] = 0
        diff += 1
    return counts.tolist()

def split_dataset(dataset, args):
    total_length = len(dataset)
    labels = torch.tensor([data.y.item() for data in dataset])
    classes = torch.unique(labels)
    class_to_indices = {}
    for cls in classes:
        cls = cls.item()
        cls_indices = (labels == cls).nonzero(as_tuple=False).view(-1).tolist()
        random.shuffle(cls_indices)
        class_to_indices[cls] = cls_indices

    # Define split ratios
    ratios = [args.unlabel_ratio, args.label_train_ratio, args.label_val_ratio, args.label_test_ratio]

    unlabel_indices = []
    label_train_indices = []
    label_val_indices = []
    label_test_indices = []

    for cls in classes:
        cls = cls.item()
        indices = class_to_indices[cls]
        n_cls = len(indices)
        counts = compute_counts(n_cls, ratios)
        n_unlabel, n_train, n_val, n_test = counts

        cls_unlabel_indices = indices[:n_unlabel]
        cls_train_indices = indices[n_unlabel:n_unlabel + n_train]
        cls_val_indices = indices[n_unlabel + n_train:n_unlabel + n_train + n_val]
        cls_test_indices = indices[n_unlabel + n_train + n_val:]

        unlabel_indices.extend(cls_unlabel_indices)
        label_train_indices.extend(cls_train_indices)
        label_val_indices.extend(cls_val_indices)
        label_test_indices.extend(cls_test_indices)

    # Create datasets
    unlabel_set = [dataset[i] for i in unlabel_indices]
    label_train_set = [dataset[i] for i in label_train_indices]
    label_val_set = [dataset[i] for i in label_val_indices]
    label_test_set = [dataset[i] for i in label_test_indices]

    # Add id attribute to each set, starting from 0
    unlabel_set = assign_ids(unlabel_set)
    label_train_set = assign_ids(label_train_set)
    label_val_set = assign_ids(label_val_set)
    label_test_set = assign_ids(label_test_set)

    return unlabel_set, label_train_set, label_val_set, label_test_set

def main():
    parser = argparse.ArgumentParser(description='Download and split dataset')
    parser.add_argument('--dataset', type=str, default="BZR", help='Name of the dataset to download')
    parser.add_argument('--download_path', type=str, default='./original_data', help='Path to download the dataset')
    parser.add_argument('--split_path', type=str, default='./splited_data', help='Path to save the splits')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--unlabel_ratio', type=float, default=0.5, help='Ratio of unlabeled data')
    parser.add_argument('--label_train_ratio', type=float, default=0.2, help='Ratio of labeled training data')
    parser.add_argument('--label_val_ratio', type=float, default=0.15, help='Ratio of labeled validation data')
    parser.add_argument('--label_test_ratio', type=float, default=0.15, help='Ratio of labeled test data')

    args = parser.parse_args()

    set_seed(args.seed)

    # Ensure that the sum of split ratios equals 1
    total_ratio = args.unlabel_ratio + args.label_train_ratio + args.label_val_ratio + args.label_test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"The sum of split ratios should be 1, but got {total_ratio}")

    # Download dataset
    dataset = download_dataset(args.dataset, args.download_path)

    # Split dataset
    unlabel_set, label_train_set, label_val_set, label_test_set = split_dataset(dataset, args)

    # Save dataset
    save_path = os.path.join(args.split_path, args.dataset)
    os.makedirs(save_path, exist_ok=True)

    torch.save(unlabel_set, os.path.join(save_path, 'unlabel_train.pt'))
    torch.save(label_train_set, os.path.join(save_path, 'train.pt'))
    torch.save(label_val_set, os.path.join(save_path, 'val.pt'))
    torch.save(label_test_set, os.path.join(save_path, 'test.pt'))

    print(f"Dataset saved to {save_path}")

    # Calculate and print the sample count and percentage for each dataset
    total_length = len(dataset)
    unlabel_len = len(unlabel_set)
    label_train_len = len(label_train_set)
    label_val_len = len(label_val_set)
    label_test_len = len(label_test_set)

    print("\nDataset Split Information:")
    print(f"Unlabeled training set (unlabel_train.pt): {unlabel_len} samples, accounting for {unlabel_len / total_length * 100:.2f}%")
    print(f"Labeled training set (train.pt): {label_train_len} samples, accounting for {label_train_len / total_length * 100:.2f}%")
    print(f"Validation set (val.pt): {label_val_len} samples, accounting for {label_val_len / total_length * 100:.2f}%")
    print(f"Test set (test.pt): {label_test_len} samples, accounting for {label_test_len / total_length * 100:.2f}%")
    print(f"Total sample count: {total_length}")
    

if __name__ == '__main__':
    main()
