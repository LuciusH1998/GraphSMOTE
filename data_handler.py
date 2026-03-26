import random
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import dgl

from dgl import RowFeatNormalizer
from datasets import *
from utils import *


class DataHandlerModule():
    def __init__(self, configuration, embeddings=None):
        self.args = argparse.Namespace(**configuration)

        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        device = torch.device(self.args.cuda_id)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        print(f"Loading and preprocessing the dataset {self.args.data_name}...")

        graph = load_data(self.args.data_name, self.args.multi_relation)
        labels = graph.ndata["y"]

        # SPLIT
        if self.args.data_name.startswith('amazon'):
            idx_unlabeled = 2013 if self.args.data_name == 'amazon_new' else 3305
            index = list(range(idx_unlabeled, len(labels)))

            idx_train, idx_rest, y_train, y_rest = train_test_split(
                index, labels[idx_unlabeled:],
                stratify=labels[idx_unlabeled:],
                train_size=self.args.train_ratio,
                random_state=self.args.seed,
                shuffle=True
            )

            idx_valid, idx_test, y_valid, y_test = train_test_split(
                idx_rest, y_rest,
                stratify=y_rest,
                test_size=self.args.test_ratio,
                random_state=self.args.seed,
                shuffle=True
            )
        else:
            index = list(range(len(labels)))

            idx_train, idx_rest, y_train, y_rest = train_test_split(
                index, labels,
                stratify=labels,
                train_size=self.args.train_ratio,
                random_state=self.args.seed,
                shuffle=True
            )

            idx_valid, idx_test, y_valid, y_test = train_test_split(
                idx_rest, y_rest,
                stratify=y_rest,
                test_size=self.args.test_ratio,
                random_state=self.args.seed,
                shuffle=True
            )

        old_num_nodes = graph.num_nodes()
        self.embeddings = embeddings

        # GRAPH SMOTE
        if getattr(self.args, "use_graph_smote", False) and getattr(self.args, "use_embedding_smote", False):
            print("Applying ADVANCED GraphSMOTE (embedding)...")
            graph, labels = self.apply_graph_smote_embedding(
                graph, labels, embeddings, idx_train, target_ratio=self.args.target_ratio
            )

        elif getattr(self.args, "use_graph_smote", False):
            print("Applying FEATURE GraphSMOTE...")
            graph, labels = self.apply_graph_smote_target(
                graph, labels, idx_train, target_ratio=self.args.target_ratio
            )

        # UPDATE TRAIN
        new_num_nodes = graph.num_nodes()
        num_new_nodes = new_num_nodes - old_num_nodes

        if num_new_nodes > 0:
            print(f"Added {num_new_nodes} synthetic nodes")

            new_node_indices = list(range(old_num_nodes, new_num_nodes))
            idx_train = list(idx_train) + new_node_indices

            if not isinstance(y_train, torch.Tensor):
                y_train = torch.tensor(y_train)

            y_train = torch.cat([
                y_train,
                torch.ones(num_new_nodes, dtype=torch.long)
            ])

        # NORMALIZATION
        if self.args.data_name.startswith('amazon'):
            transform = RowFeatNormalizer(subtract_min=True, node_feat_names=['x'])
            graph = transform(graph)

        graph.ndata["x"] = torch.FloatTensor(graph.ndata["x"]).contiguous()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.args.emb_size))

        valid_loader = dgl.dataloading.DataLoader(
            graph, idx_valid, sampler,
            batch_size=self.args.batch_size,
            shuffle=False, drop_last=False, use_uva=True
        )

        test_loader = dgl.dataloading.DataLoader(
            graph, idx_test, sampler,
            batch_size=self.args.batch_size,
            shuffle=False, drop_last=False, use_uva=True
        )

        self.dataset = {
            'features': graph.ndata["x"],
            'labels': labels,
            'graph': graph,
            'idx_train': idx_train,
            'idx_valid': idx_valid,
            'idx_test': idx_test,
            'y_train': y_train,
            'y_valid': y_valid,
            'y_test': y_test,
            'train_loader': None,
            'valid_loader': valid_loader,
            'test_loader': test_loader,
            'idx_total': list(range(len(labels)))
        }

        print("Finished data loading and preprocessing!")

    def project_embeddings_to_features(self, emb, original_x):
        W = torch.linalg.lstsq(emb, original_x).solution
        return emb @ W

    def apply_graph_smote_embedding(self, graph, labels, embeddings, train_idx,
                                   target_ratio=0.05, minority_class=1, k=5):

        if embeddings is None:
            print("No embeddings provided → skipping")
            return graph, labels

        x = F.normalize(embeddings, dim=1)
        y = labels

        train_idx_tensor = torch.tensor(train_idx)
        fraud_idx = train_idx_tensor[y[train_idx_tensor] == minority_class]

        if len(fraud_idx) < 2:
            return graph, labels

        total_nodes = len(train_idx)
        N_fraud = len(fraud_idx)

        N_new = int((target_ratio * total_nodes - N_fraud) / (1 - target_ratio))
        N_new = min(N_new, 10 * N_fraud)

        dist = torch.cdist(x[fraud_idx], x[fraud_idx])
        knn = dist.topk(k=min(k+1, len(fraud_idx)), largest=False).indices[:, 1:]

        new_embeddings = []
        new_edges_src = []
        new_edges_dst = []

        for _ in range(N_new):
            i = torch.randint(0, len(fraud_idx), (1,))
            src = fraud_idx[i]

            neighbor = fraud_idx[knn[i][torch.randint(0, knn.shape[1], (1,))]]

            alpha = torch.rand(1)
            new_emb = x[src] + alpha * (x[neighbor] - x[src])
            new_embeddings.append(new_emb.squeeze())

            new_node_id = graph.num_nodes() + len(new_embeddings) - 1

            for n in knn[i]:
                new_edges_src.append(n.item())
                new_edges_dst.append(new_node_id)

        new_embeddings = torch.stack(new_embeddings)

        new_features = self.project_embeddings_to_features(
            new_embeddings,
            graph.ndata['x']
        )

        graph.add_nodes(len(new_features))
        graph.ndata['x'] = torch.cat([graph.ndata['x'], new_features], dim=0)

        new_labels = torch.ones(len(new_features), dtype=y.dtype)
        labels = torch.cat([labels, new_labels], dim=0)
        graph.ndata['y'] = labels

        for etype in graph.etypes:
            graph.add_edges(new_edges_src, new_edges_dst, etype=etype)

        return graph, labels

    def apply_graph_smote_target(self, graph, labels, train_idx,
                                target_ratio=0.05, minority_class=1, k=5):

        x = graph.ndata['x']
        y = labels

        train_idx_tensor = torch.tensor(train_idx)
        fraud_idx = train_idx_tensor[y[train_idx_tensor] == minority_class]

        if len(fraud_idx) < 2:
            return graph, labels

        total_nodes = len(train_idx)
        N_fraud = len(fraud_idx)

        N_new = int((target_ratio * total_nodes - N_fraud) / (1 - target_ratio))
        N_new = min(N_new, 10 * N_fraud)

        dist = torch.cdist(x[fraud_idx], x[fraud_idx])
        knn = dist.topk(k=min(k+1, len(fraud_idx)), largest=False).indices[:, 1:]

        new_features = []
        new_edges_src = []
        new_edges_dst = []

        for _ in range(N_new):
            i = random.randint(0, len(fraud_idx) - 1)
            src = fraud_idx[i]
            neighbor = fraud_idx[random.choice(knn[i]).item()]

            alpha = torch.rand(1)
            new_feat = x[src] + alpha * (x[neighbor] - x[src])

            new_features.append(new_feat)

            new_node_id = graph.num_nodes() + len(new_features) - 1
            new_edges_src += [src.item(), neighbor.item()]
            new_edges_dst += [new_node_id, new_node_id]

        new_features = torch.stack(new_features)

        graph.add_nodes(len(new_features))
        graph.ndata['x'] = torch.cat([graph.ndata['x'], new_features], dim=0)

        new_labels = torch.ones(len(new_features), dtype=y.dtype)
        labels = torch.cat([labels, new_labels], dim=0)
        graph.ndata['y'] = labels

        for etype in graph.etypes:
            graph.add_edges(new_edges_src, new_edges_dst, etype=etype)

        return graph, labels