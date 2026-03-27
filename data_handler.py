
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

        if getattr(self.args, "use_graph_smote", False) and getattr(self.args, "use_embedding_smote", False):
            print("Applying ADVANCED GraphSMOTE (embedding)...")
            graph, labels = self.apply_graph_smote_embedding(
                graph, labels, embeddings, idx_train, target_ratio=self.args.target_ratio
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