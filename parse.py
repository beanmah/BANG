import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset and paths
    parser.add_argument("--dataset", type=str, default="BZR",
                        help="Choose a dataset")
    parser.add_argument("--splited_data_path", type=str, default="splited_data",
                        help="Path to splited data")
    parser.add_argument("--semi_supervised_data_path", type=str, default="semi_supervised_data",
                        help="Output path for semi-supervised data")
    parser.add_argument("--model_path", type=str, default="saved_models",
                        help="Path to saved models")
    parser.add_argument("--pretrained_gnn_encoder_path", type=str, default="models_gnn_encoder",
                        help="Path to pretrained GNN encoder models")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path to pretrained models")

    # Model architecture
    parser.add_argument("--n_hidden", type=int, default=128,
                        help="Number of hidden units")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=31,
                        help="Random seed")

    # Graph construction
    parser.add_argument("--knn_nei_num", type=int, default=5,
                        help="Number of nearest neighbors for KNN graph")
    parser.add_argument("--knn_method", type=str, default="combined",
                        choices=["kernel", "embedding", "combined"],
                        help="Method to construct KNN graph")

    # Training mode
    parser.add_argument("--training_mode", type=str, default="supervised",
                        choices=["supervised", "semi_supervised"],
                        help="Training mode: supervised or semi_supervised")

    # Data augmentation
    parser.add_argument("--use_augmentation", type=int, default=1,
                        help="Use data augmentation (0: False, 1: True)")
    parser.add_argument("--augmentation_ratio", type=float, default=0.5,
                        help="Ratio of augmented samples to original samples")
    parser.add_argument("--label_aug_proportions", type=str, default=None,
                        help="Label augmentation proportions in format 'label1:prop1,label2:prop2'")

    return parser.parse_args()