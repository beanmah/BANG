# dataset.py

import os
import os.path as osp
import torch
import shutil
from utils import *

from typing import Optional, Callable, List
from torch.utils.data import Dataset as BaseDataset

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import subgraph, degree, add_remaining_self_loops
from torch_geometric.data.collate import collate



class Dataset(BaseDataset):
    def __init__(self, dataset, all_dataset, kernel_idx, knn_edge_index):
        self.dataset = dataset
        self.all_dataset = all_dataset
        self.kernel_idx = kernel_idx
        self.knn_edge_index = knn_edge_index

    def _get_feed_dict(self, index):
        feed_dict = self.dataset[index]

        return feed_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_feed_dict(index)


    def collate_batch(self, feed_dicts):
        batch_id = torch.tensor([feed_dict.id for feed_dict in feed_dicts])
        # prevent testing data leakage
        idx = torch.arange(batch_id.shape[0])

        # add_knn_dataset to feed_dicts
        pad_knn_id = find_knn_id(batch_id, self.kernel_idx)
        feed_dicts.extend([self.all_dataset[i] for i in pad_knn_id])

        data, slices, _ = collate(
            feed_dicts[0].__class__,
            data_list=feed_dicts,
            increment=True,
            add_batch=True,
        )

        knn_edge_index, _ = subgraph(data.id, self.knn_edge_index, relabel_nodes=True)

        knn_edge_index, _ = add_remaining_self_loops(knn_edge_index)
        row, col = knn_edge_index
        knn_deg = degree(col, data.id.shape[0])
        deg_inv_sqrt = knn_deg.pow(-0.5)
        edge_weight = deg_inv_sqrt[col] * deg_inv_sqrt[col]

        knn_edge_index = knn_edge_index.long()

        batch = {
            'data': data,
            'idx': idx,
            'knn_edge_index': knn_edge_index,
            'knn_edge_weight': edge_weight
        }

        return batch

    


class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)

    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        self.data, self.slices, self.sizes = out

        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
        
        self.data.id = torch.arange(0, self.data.y.size(0))
        self.slices['id'] = self.slices['y'].clone()


    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self.data, self.slices, sizes), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'



def load_data(split_path):
    """
    Load the pre-split data from the split directory and compute dataset statistics.
    
    Args:
        dataset_name (str): The name of the dataset.
        split_path (str): The path to the split data.
    
    Returns:
        train_data, val_data, test_data: The loaded datasets.
        n_feat: Number of features in the dataset.
        n_class: Number of classes in the dataset.
    """
    train_data = torch.load(os.path.join(split_path, 'train.pt'))
    val_data = torch.load(os.path.join(split_path, 'val.pt'))
    test_data = torch.load(os.path.join(split_path, 'test.pt'))
    
    # Combine all datasets to calculate feature and class counts
    all_data = train_data + val_data + test_data
    
    # Calculate number of features
    if len(all_data) > 0:
        n_feat = all_data[0].num_features
    else:
        n_feat = 0
    
    # Calculate number of classes
    if all(data.y is not None for data in all_data):
        n_class = max(data.y.item() for data in all_data) + 1
    else:
        n_class = 0
    
    return train_data, val_data, test_data, n_feat, n_class