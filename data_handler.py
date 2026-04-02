import random
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import dgl

from datasets import *
from utils import *


class DataHandlerModule():
    def __init__(self, configuration, embeddings=None):
        self.args = argparse.Namespace(**configuration)

        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        cuda_id = self.args.cuda_id
        device = torch.device(f'cuda:{cuda_id}' if isinstance(cuda_id, int) else cuda_id)

        if device.type == "cuda":
            torch.cuda.set_device(device)

        print(f"Loading and preprocessing the dataset {self.args.data_name}...")

        graph = load_data(
            self.args.data_name,
            self.args.multi_relation,
            raw_dir=self.args.raw_dir,
            sample=self.args.sample,
            seed=self.args.seed
        )

        labels = graph.ndata["y"]

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

        # =========================
        # APPLY GRAPHSMOTE
        # =========================
        if getattr(self.args, "use_graph_smote", False) and getattr(self.args, "use_embedding_smote", False):
            print("Applying ADVANCED GraphSMOTE (embedding)...")
            graph, labels = self.apply_graph_smote_embedding(
                graph, labels, embeddings, idx_train,
                target_ratio=self.args.target_ratio
            )

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

    # =========================
    # GRAPHSMOTE (FIXED)
    # =========================
    def apply_graph_smote_embedding(self, graph, labels, embeddings, train_idx,
                                   target_ratio=0.05, minority_class=1, k=5):

        if embeddings is None:
            print("No embeddings provided → skipping")
            return graph, labels

        x = F.normalize(embeddings, dim=1)
        y = labels

        train_idx_tensor = torch.tensor(train_idx)

        # Minority nodes in TRAIN ONLY (correct practice)
        fraud_idx = train_idx_tensor[y[train_idx_tensor] == minority_class]

        if len(fraud_idx) < 2:
            print("Not enough minority samples → skipping SMOTE")
            return graph, labels

        # =========================
        # FIX 1: CLEAR TARGET CALCULATION
        # =========================
        N_train = len(train_idx)
        N_fraud = len(fraud_idx)

        target_fraud_train = int(target_ratio * N_train)
        N_new = max(0, target_fraud_train - N_fraud)

        # Safety cap
        N_new = min(N_new, 10 * N_fraud)

        print(f"[GraphSMOTE] Train nodes: {N_train}")
        print(f"[GraphSMOTE] Fraud (train): {N_fraud}")
        print(f"[GraphSMOTE] Target fraud: {target_fraud_train}")
        print(f"[GraphSMOTE] Generating: {N_new} synthetic nodes")

        if N_new == 0:
            return graph, labels

        # =========================
        # KNN IN EMBEDDING SPACE
        # =========================
        dist = torch.cdist(x[fraud_idx], x[fraud_idx])
        knn = dist.topk(k=min(k + 1, len(fraud_idx)), largest=False).indices[:, 1:]

        new_features = []
        new_edges_src = []
        new_edges_dst = []

        old_features = graph.ndata['x']

        for _ in range(N_new):
            i = random.randint(0, len(fraud_idx) - 1)
            src = fraud_idx[i]

            j = random.randint(0, knn.shape[1] - 1)
            neighbor = fraud_idx[knn[i][j]]

            alpha = torch.rand(1)

            # =========================
            # FIX 2: FEATURE INTERPOLATION (CORRECT)
            # =========================
            src_feat = old_features[src]
            neigh_feat = old_features[neighbor]

            new_feat = src_feat + alpha * (neigh_feat - src_feat)
            new_features.append(new_feat)

            new_node_id = graph.num_nodes() + len(new_features) - 1

            for n in knn[i]:
                new_edges_src.append(n.item())
                new_edges_dst.append(new_node_id)

        new_features = torch.stack(new_features)

        # =========================
        # ADD NODES
        # =========================
        num_new = new_features.shape[0]

        graph.add_nodes(num_new)

        graph.ndata['x'] = torch.cat([old_features, new_features], dim=0)

        new_labels = torch.ones(num_new, dtype=y.dtype)
        graph.ndata['y'] = torch.cat([labels, new_labels], dim=0)

        labels = graph.ndata['y']

        for etype in graph.etypes:
            graph.add_edges(new_edges_src, new_edges_dst, etype=etype)

        return graph, labels