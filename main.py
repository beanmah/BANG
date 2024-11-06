
# main.py

import os
from parse import parse_args
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *
from model import *
from learn import train_supervised, train_semi_supervised, eval
from dataset import load_data, Dataset
from construct_knn_graph import build_knn_graph_from_kernel, build_knn_graph_from_embeddings, build_knn_graph_combined
import torch
import json



def run(args):

    train_data, val_data, test_data, args.n_feat, args.n_class = load_data(args.data_path)
    dataset_all = train_data + val_data + test_data
    dataset_train_and_val = train_data + val_data
    
    if args.knn_method == 'kernel':
        _, kernel_idx_train_and_val, knn_edge_index_train_and_val = build_knn_graph_from_kernel(dataset_train_and_val, args)
        _, kernel_idx_test, knn_edge_index_test = build_knn_graph_from_kernel(dataset_all, args)
    elif args.knn_method == 'embedding':
        _, kernel_idx_train_and_val, knn_edge_index_train_and_val = build_knn_graph_from_embeddings(dataset_train_and_val, args)
        _, kernel_idx_test, knn_edge_index_test = build_knn_graph_from_embeddings(dataset_all, args)
    elif args.knn_method == 'combined':
        _, kernel_idx_train_and_val, knn_edge_index_train_and_val = build_knn_graph_combined(dataset_train_and_val, args)
        _, kernel_idx_test, knn_edge_index_test = build_knn_graph_combined(dataset_all, args)
    else:
        raise ValueError(f"Unknown KNN method: {args.knn_method}")

    train_dataset = Dataset(train_data, dataset_train_and_val, kernel_idx_train_and_val, knn_edge_index_train_and_val)
    val_dataset = Dataset(val_data, dataset_train_and_val, kernel_idx_train_and_val, knn_edge_index_train_and_val)
    test_dataset = Dataset(test_data, dataset_all, kernel_idx_test, knn_edge_index_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_batch)

    encoder = GIN(args).to(args.device)
    gnn = GCN(args).to(args.device)
    classifier = MLP_Classifier(args).to(args.device)

    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_g = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_c = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc = 0.0
    best_model_state = {}

    epoch_bar = tqdm(range(args.epochs), desc=f"Training Progress", unit="epoch")

    for epoch in epoch_bar:
        encoder.train()
        gnn.train()
        classifier.train()

        if args.training_mode == 'supervised':
            train_result = train_supervised(encoder, gnn, classifier, train_loader, optimizer_e, optimizer_g, optimizer_c, args)
        elif args.training_mode == 'semi_supervised':
            train_result = train_semi_supervised(encoder, gnn, classifier, train_loader, optimizer_e, optimizer_g, optimizer_c, args)
        else:
            raise ValueError(f"Unknown training mode: {args.training_mode}")

        train_loss = train_result['loss']
        train_acc = train_result['acc']
        train_auc = train_result['auc']

        encoder.eval()
        gnn.eval()
        classifier.eval()
        with torch.no_grad():
            val_result = eval(encoder, gnn, classifier, val_loader, args)
        val_loss = val_result.get('loss', 0.0)
        val_acc = val_result['acc']
        val_auc = val_result['auc']

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {
                'encoder_state_dict': encoder.state_dict(),
                'gnn_state_dict': gnn.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_e_state_dict': optimizer_e.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_c_state_dict': optimizer_c.state_dict(),
            }

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.4f}",
            train_auc=f"{train_auc:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            val_auc=f"{val_auc:.4f}"
        )

    if best_model_state:
        encoder.load_state_dict(best_model_state['encoder_state_dict'])
        gnn.load_state_dict(best_model_state['gnn_state_dict'])
        classifier.load_state_dict(best_model_state['classifier_state_dict'])

    encoder.eval()
    gnn.eval()
    classifier.eval()
    with torch.no_grad():
        test_eval = eval(encoder, gnn, classifier, test_loader, args)

    print(f"Test Acc = {test_eval['acc']:.4f}, Test Auc = {test_eval['auc']:.4f}")

    model_save_dir = os.path.join(args.path, 'saved_models')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'model_{args.dataset}.pth')
    torch.save(best_model_state, model_save_path)
    print(f"Saved the best model at {model_save_path}")

    return test_eval['acc'], test_eval['auc']


if __name__ == '__main__':

    args = parse_args()
    set_seed(args.seed)
    args.path = os.path.dirname(os.path.abspath(__file__))


    

    if args.training_mode == 'supervised':
        args.data_path = os.path.join(args.path, 'splited_data', args.dataset)
    elif args.training_mode == 'semi_supervised':
        args.data_path = os.path.join(args.path, 'semi_supervised_data', args.dataset)
    else:
        raise ValueError(f"Unknown training mode: {args.training_mode}")

    Acc, Auc = run(args)

    print("-" * 50)
    print(f'Test Acc: {Acc:.6f}')
    print(f'Test Auc: {Auc:.6f}')
    print("-" * 50)


    if args.training_mode == 'semi_supervised':

        result_dir = os.path.join("result", args.dataset)
        os.makedirs(result_dir, exist_ok=True)
        result_file = os.path.join(result_dir, 'results.jsonl')

        result_data = {
            "Dataset": args.dataset,
            "Seed": args.seed,
            "Test Acc": round(Acc, 6),
            "Test Auc": round(Auc, 6)
        }
        
        with open(result_file, 'a') as f:
            f.write(json.dumps(result_data) + '\n')

        print(f"Results saved to {result_file}")
