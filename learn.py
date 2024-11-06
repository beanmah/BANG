
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score
import numpy as np
import random
from utils import *


def split_data(data):
    ptr = data.ptr  # Node separation pointer
    num_graphs = len(ptr) - 1  # Number of graphs

    data_list = []

    for i in range(num_graphs):
        start, end = ptr[i], ptr[i + 1]

        # Extract node features and edge information for the graph
        sub_x = data.x[start:end]
        edge_mask = (data.edge_index[0] >= start) & (data.edge_index[0] < end)
        sub_edge_index = data.edge_index[:, edge_mask] - start

        # Get label information, set default value if not exists
        sub_y = data.y[i]
        sub_predicted_label = data.predicted_label[i]
        sub_confidence = data.label_confidence[i]
        sub_is_pseudo_label = data.is_pseudo_label[i]

        # Create new Data object
        sub_data = Data(
            x=sub_x,
            edge_index=sub_edge_index,
            y=sub_y,
            predicted_label=sub_predicted_label,
            label_confidence=sub_confidence,
            is_pseudo_label=sub_is_pseudo_label
        )

        data_list.append(sub_data)

    return data_list


def augment_data(core_data, data_list, augmentation_ratio, label_aug_proportions=None):
    # Group core data by labels
    label_to_data = {}
    for data in core_data:
        if data.is_pseudo_label == 0:
            label = int(data.y.item())  # Convert tensor to int
        else:
            label = int(data.predicted_label.item())  # Convert tensor to int
        if label not in label_to_data:
            label_to_data[label] = []
        label_to_data[label].append(data)
    
    total_samples = len(core_data)
    total_aug_samples = int(total_samples * augmentation_ratio)
    
    # Calculate proportion of each class in core data
    label_counts = {label: len(datas) for label, datas in label_to_data.items()}
    label_ratios = {label: count / total_samples for label, count in label_counts.items()}
    
    if label_aug_proportions is None:
        # Calculate augmentation sample count for each class, allocate based on inverse class ratio
        inverse_ratios = {label: 1 / ratio if ratio > 0 else 0 for label, ratio in label_ratios.items()}
        total_inverse = sum(inverse_ratios.values())
        label_aug_ratios = {label: inverse_ratios[label] / total_inverse for label in inverse_ratios}
    else:
        # Use specified augmentation proportions
        label_aug_ratios = {}
        for label in label_to_data.keys():
            label_aug_ratios[label] = label_aug_proportions.get(label, 0)
        total_specified = sum(label_aug_ratios.values())
        if total_specified == 0:
            # If total is 0, distribute evenly
            num_labels = len(label_aug_ratios)
            label_aug_ratios = {label: 1 / num_labels for label in label_aug_ratios}
        else:
            # Normalize
            label_aug_ratios = {label: ratio / total_specified for label, ratio in label_aug_ratios.items()}
    

    # Calculate initial floating-point augmentation samples
    label_aug_counts_float = {label: total_aug_samples * label_aug_ratios[label] for label in label_aug_ratios}
    
    # Split into integer and decimal parts
    label_aug_counts_int = {label: int(count) for label, count in label_aug_counts_float.items()}
    label_aug_counts_remainders = {label: count - int(count) for label, count in label_aug_counts_float.items()}
    
    # Calculate assigned and remaining samples
    total_assigned = sum(label_aug_counts_int.values())
    samples_remaining = total_aug_samples - total_assigned
    
    # Sort by decimal part in descending order
    sorted_labels = sorted(label_aug_counts_remainders.items(), key=lambda x: x[1], reverse=True)
    
    # Allocate remaining samples
    for i in range(samples_remaining):
        label = sorted_labels[i][0]
        label_aug_counts_int[label] += 1
    
    # Use final augmentation sample counts
    label_aug_counts = label_aug_counts_int
    
    augmented_data = []
    for label, count in label_aug_counts.items():
        datas = label_to_data[label]
        if len(datas) < 2:
            continue  # Skip if less than 2 graphs in the class
        for _ in range(count):
            # Randomly select two graphs
            idx1, idx2 = random.sample(range(len(datas)), 2)
            g1, g2 = datas[idx1], datas[idx2]

            # Check node counts
            if g1.x.size(0) == 0 or g2.x.size(0) == 0:
                continue  # Skip graphs with no nodes

            # Get indices of g1 and g2 in data_list
            idx_g1 = data_list.index(g1)
            idx_g2 = data_list.index(g2)
            # Extract subgraphs from each graph
            sub_g1 = extract_subgraph(g1)
            sub_g2 = extract_subgraph(g2)

            # Check subgraph node counts
            if sub_g1.x.size(0) == 0 or sub_g2.x.size(0) == 0:
                continue  # Skip subgraphs with no nodes

            # Combine subgraphs
            new_data = combine_subgraphs(sub_g1, sub_g2)
            # Get device information
            device = new_data.x.device  # Ensure new tensor is on correct device
            # Handle labels and confidence
            if g1.is_pseudo_label == 0 and g2.is_pseudo_label == 0:
                new_data.y = g1.y.clone().to(device)
                new_data.label_confidence = torch.tensor(1.0, dtype=torch.float, device=device)
                new_data.predicted_label = torch.tensor(-1, dtype=torch.long, device=device)
                new_data.is_pseudo_label = torch.tensor(0, dtype=torch.long, device=device)
            elif g1.is_pseudo_label == 1 and g2.is_pseudo_label == 1:
                new_data.y = g1.y.clone().to(device)
                new_data.predicted_label = g1.predicted_label.clone().to(device)
                avg_confidence = (g1.label_confidence + g2.label_confidence) / 2
                new_data.label_confidence = avg_confidence.clone().detach().to(device)
                new_data.is_pseudo_label = torch.tensor(1, dtype=torch.long, device=device)
            else:
                # One true label, one pseudo label
                if g1.is_pseudo_label == 0:
                    new_data.y = g1.y.clone().to(device)
                    new_data.predicted_label = g1.y.clone().to(device)
                else:
                    new_data.y = g2.y.clone().to(device)
                    new_data.predicted_label = g2.y.clone().to(device)
                avg_confidence = (g1.label_confidence + g2.label_confidence) / 2
                new_data.label_confidence = avg_confidence.clone().detach().to(device)
                new_data.is_pseudo_label = torch.tensor(1, dtype=torch.long, device=device)
            # Record parent graph indices
            new_data.parent_indices = [idx_g1, idx_g2]
            augmented_data.append(new_data)
    
    return augmented_data




def extract_subgraph(data):
    num_nodes = data.x.size(0)  # Get number of nodes
    device = data.edge_index.device  # Get device of edge_index

    # Randomly select a portion of nodes
    sub_num_nodes = random.randint(int(0.4 * num_nodes), int(0.6 * num_nodes))
    node_indices = random.sample(range(num_nodes), sub_num_nodes)
    node_indices = torch.tensor(node_indices, dtype=torch.long, device=device)  # Move node_indices to correct device

    # Extract node features for subgraph
    sub_x = data.x[node_indices]

    # Extract edges for subgraph
    edge_index = data.edge_index
    # Create a set of node indices for quick lookup
    node_set = set(node_indices.cpu().numpy())
    # Find edges where both endpoints are in selected node set
    mask = (torch.isin(edge_index[0], node_indices) & torch.isin(edge_index[1], node_indices))
    sub_edge_index = edge_index[:, mask]

    # Remap node indices
    idx_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices.cpu())}
    sub_edge_index = sub_edge_index.clone()
    sub_edge_index[0] = torch.tensor([idx_mapping[idx.item()] for idx in sub_edge_index[0].cpu()], device=device)
    sub_edge_index[1] = torch.tensor([idx_mapping[idx.item()] for idx in sub_edge_index[1].cpu()], device=device)

    # Create new Data object
    sub_data = Data(
        x=sub_x,
        edge_index=sub_edge_index,
        y=data.y.clone(),
        predicted_label=data.predicted_label.clone(),
        label_confidence=data.label_confidence.clone(),
        is_pseudo_label=data.is_pseudo_label.clone()
    )

    return sub_data


def combine_subgraphs(g1, g2):
    # Combine node features
    new_x = torch.cat([g1.x, g2.x], dim=0)
    
    # Combine edge indices
    g2_edge_index = g2.edge_index + g1.x.size(0)
    new_edge_index = torch.cat([g1.edge_index, g2_edge_index], dim=1)
    
    # Get node count for each subgraph
    num_nodes_g1 = g1.x.size(0)
    num_nodes_g2 = g2.x.size(0)
    
    # Check if node counts are sufficient
    if num_nodes_g1 == 0 or num_nodes_g2 == 0:
        # If either subgraph has no nodes, return merged graph without cross-subgraph edges
        new_data = Data(
            x=new_x,
            edge_index=new_edge_index
        )
        return new_data

    # Calculate maximum possible connections
    max_num_connect = min(num_nodes_g1, num_nodes_g2)
    
    # Check max connections, prevent num_connect from exceeding node count
    if max_num_connect < 2:
        # If max connections less than 2, return without adding cross-subgraph edges
        return Data(x=new_x, edge_index=new_edge_index)

    # Random connect partial nodes
    num_connect = max(2, int(0.1 * min(g1.x.size(0), g2.x.size(0))))

    g1_indices = random.sample(range(g1.x.size(0)), num_connect)
    g2_indices = random.sample(range(g2.x.size(0)), num_connect)
    g2_indices = [idx + g1.x.size(0) for idx in g2_indices]
    
    # Create cross-subgraph edges
    cross_edges = []
    for idx1 in g1_indices:
        for idx2 in g2_indices:
            cross_edges.append([idx1, idx2])
            cross_edges.append([idx2, idx1])
    
    if cross_edges:
        device = new_edge_index.device
        cross_edge_index = torch.tensor(cross_edges, dtype=torch.long, device=device).t()
        new_edge_index = torch.cat([new_edge_index, cross_edge_index], dim=1)
    
    # Create new Data object
    new_data = Data(
        x=new_x,
        edge_index=new_edge_index
        # Don't set y, predicted_label, etc.
    )
    return new_data


def recalculate_knn_edge_index(original_knn_edge_index, data_list, augmented_data, args):
    """
    Recalculate knn_edge_index to include edges for augmented data.

    Args:
    - original_knn_edge_index: Original knn_edge_index based on batch's original data.
    - data_list: Complete list containing original and augmented data.
    - augmented_data: List of augmented data.
    - args: Namespace containing necessary parameters.

    Returns:
    - combined_knn_edge_index: Updated knn_edge_index including edges for augmented data.
    """
    device = args.device
    num_total = len(data_list)
    num_original = num_total - len(augmented_data)  # Number of original graphs

    # No need to adjust indices in original knn_edge_index as they still correspond to original data positions
    adjusted_knn_edge_index = original_knn_edge_index.clone()

    # Build new edge list
    edge_index_list = [adjusted_knn_edge_index]

    # Add edges between augmented graphs and their parent graphs
    for idx_augmented, augmented_graph in enumerate(augmented_data):
        augmented_idx = num_original + idx_augmented  # Index of augmented graph in data_list
        parent_indices = augmented_graph.parent_indices  # [idx_g1, idx_g2]

        # Add edges between augmented graph and parent graphs
        for parent_idx in parent_indices:
            edge_index = torch.tensor([[augmented_idx, parent_idx],
                                       [parent_idx, augmented_idx]], dtype=torch.long, device=device)
            edge_index_list.append(edge_index)

    combined_knn_edge_index = torch.cat(edge_index_list, dim=1)

    return combined_knn_edge_index


def train_semi_supervised(encoder, gnn, classifier, data_loader, optimizer_e, optimizer_g, optimizer_c, args):
    
    encoder.train()
    gnn.train()
    classifier.train()

    total_loss = 0
    pred = []
    truth = []
    all_probs = []
    all_labels = []

    for i, batch in enumerate(data_loader):
        data = batch['data'].to(args.device)
        data_idx = batch['idx'].to(args.device)
        knn_edge_index = batch['knn_edge_index'].to(args.device)

        data_list = split_data(data)
        core_data = [data_list[i] for i in data_idx.tolist()]

        if args.use_augmentation:
            # augmented_data = augment_data(core_data, args.augmentation_ratio, data_list)
            if args.label_aug_proportions and args.label_aug_proportions != "None":
                label_aug_proportions = parse_label_aug_proportions(args.label_aug_proportions)
            else:
                label_aug_proportions = None
            augmented_data = augment_data(core_data, data_list, args.augmentation_ratio, label_aug_proportions)

            data_list.extend(augmented_data)
            augmented_indices = list(range(len(data_list) - len(augmented_data), len(data_list)))
        else:
            augmented_indices = []

        combined_data = Batch.from_data_list(data_list).to(args.device)

        H = encoder(combined_data.x, combined_data.edge_index, combined_data.ptr)
        logits = classifier(H)

        batch_loss = 0.0
        valid_samples = 0

        all_indices = data_idx.tolist() + augmented_indices


        for sample_idx in all_indices:
            sample_data = data_list[sample_idx]
            logit = logits[sample_idx]

            is_pseudo_label = sample_data.is_pseudo_label.item()
            if is_pseudo_label == 1:
                label = sample_data.predicted_label.to(args.device)
                confidence = sample_data.label_confidence.to(args.device)
                loss = F.cross_entropy(logit.unsqueeze(0), label.unsqueeze(0), reduction='none')
                weighted_loss = loss * confidence
                batch_loss += weighted_loss.mean()
                valid_samples += confidence.item()
                truth.append(label.item())
            elif is_pseudo_label == 0:
                label = sample_data.y.to(args.device)
                loss = F.cross_entropy(logit.unsqueeze(0), label.unsqueeze(0))
                batch_loss += loss
                valid_samples += 1.0
                truth.append(label.item())
            else:
                raise ValueError(f"Sample {sample_idx} has neither true label nor predicted label.")


            pred_label = logit.argmax(dim=-1).item()
            pred.append(pred_label)
            prob = torch.softmax(logit, dim=-1)[1].item()
            all_probs.append(prob)
            all_labels.append(truth[-1])

        if valid_samples > 0:
            batch_loss = batch_loss / valid_samples
            total_loss += batch_loss.item()

            optimizer_e.zero_grad()
            optimizer_g.zero_grad()
            optimizer_c.zero_grad()
            batch_loss.backward()
            optimizer_e.step()
            optimizer_g.step()
            optimizer_c.step()
        else:
            continue 

    if len(pred) > 0:
        avg_loss = total_loss / (i + 1)
        acc = (np.array(pred) == np.array(truth)).mean()
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    else:
        avg_loss = 0.0
        acc = 0.0
        auc = 0.5

    return {
        'loss': avg_loss,
        'acc': acc,
        'auc': auc
    }


def train_supervised(encoder, gnn, classifier, data_loader, optimizer_e, optimizer_g, optimizer_c, args):
    encoder.train()
    gnn.train()
    classifier.train()

    total_loss = 0
    pred = []
    truth = []
    all_probs = []
    all_labels = []

    for i, batch in enumerate(data_loader):
        data = batch['data'].to(args.device)
        data_idx = batch['idx'].to(args.device)
        knn_edge_index = batch['knn_edge_index'].to(args.device)

        H = encoder(data.x, data.adj_t, data.ptr)
        logits = classifier(H)[data_idx] 

        true_labels = data.y[data_idx] 
        true_labels = true_labels.to(torch.long)

        loss = F.cross_entropy(logits, true_labels)
        total_loss += loss.item()

        optimizer_e.zero_grad()
        optimizer_g.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer_e.step()
        optimizer_g.step()
        optimizer_c.step()

        pred_labels = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)[:, 1]

        labels_np = true_labels.cpu().numpy()
        preds_np = pred_labels.cpu().numpy()
        probs_np = probs.detach().cpu().numpy()


        pred.extend(preds_np)
        truth.extend(labels_np)
        all_probs.extend(probs_np)
        all_labels.extend(labels_np)

    avg_loss = total_loss / (i + 1)
    acc = (np.array(pred) == np.array(truth)).mean()
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

    return {
        'loss': avg_loss,
        'acc': acc,
        'auc': auc
    }




def eval(encoder, gnn, classifier, data_loader, args):
    encoder.eval()
    gnn.eval()
    classifier.eval()

    pred, truth = [], []
    total_loss = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            data = batch['data'].to(args.device)
            data_idx = batch['idx'].to(args.device)
            knn_edge_index = batch['knn_edge_index'].to(args.device)

            H = encoder(data.x, data.adj_t, data.ptr)
            logits = classifier(H)[data_idx]

            true_labels = data.y[data_idx]
            true_labels = true_labels.to(torch.long)

            loss = F.cross_entropy(logits, true_labels, reduction='mean')
            total_loss += loss.item()

            pred_labels = logits.argmax(dim=-1)
            probs = torch.softmax(logits, dim=-1)[:, 1]

            labels_np = true_labels.cpu().numpy()
            preds_np = pred_labels.cpu().numpy()
            probs_np = probs.cpu().numpy()

            pred.extend(preds_np)
            truth.extend(labels_np)
            all_probs.extend(probs_np)
            all_labels.extend(labels_np)

    avg_loss = total_loss / (i + 1)
    acc = (np.array(pred) == np.array(truth)).mean()
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

    return {
        'loss': avg_loss,
        'acc': acc,
        'auc': auc
    }
