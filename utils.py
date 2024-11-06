import numpy as np
import random
import torch
import os



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_class_num(imb_ratio, num_train, num_val):
    c_train_num = [int(imb_ratio * num_train), num_train -
                   int(imb_ratio * num_train)]

    c_val_num = [int(imb_ratio * num_val), num_val - int(imb_ratio * num_val)]

    return c_train_num, c_val_num


def find_knn_id(batch_id, kernel_idx):
    knn_id = set(kernel_idx[batch_id].view(-1).tolist())
    pad_knn_id = knn_id.difference(set(batch_id.tolist()))
    
    return sorted(list(pad_knn_id)) 


def parse_label_aug_proportions(s):
    label_aug_proportions = {}
    for item in s.split(','):
        label_str, prop_str = item.split(':')
        label = int(label_str.strip())
        proportion = float(prop_str.strip())
        label_aug_proportions[label] = proportion
    return label_aug_proportions