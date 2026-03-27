import os
import time
import random
import argparse
from typing import Tuple

import torch
import numpy as np
import dgl
import torch.nn as nn

from models import *
from datasets import *
from data_handler import DataHandlerModule
from result_manager import ResultManager
from utils import test, generate_batch_idx

PYG_DIR_PATH = "./data/pyg"


class ModelHandlerModule():
    def __init__(self, configuration, datahandler: DataHandlerModule):
        self.args = argparse.Namespace(**configuration)
        self.dataset = datahandler.dataset
        self.epochs = self.args.epochs
        self.patience = self.args.patience
        self.result = ResultManager(args=configuration)

        # =========================
        # FIXED DEVICE HANDLING
        # =========================
        self.seed = self.args.seed
        cuda_id = self.args.cuda_id
        device = torch.device(f'cuda:{cuda_id}' if isinstance(cuda_id, int) else cuda_id)

        if device.type == "cuda":
            torch.cuda.set_device(device)

        self.model = self.select_model()
        self.model.cuda()

    # =========================
    # SEED SETTING
    # =========================
    def set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(self.seed)

    # =========================
    # MODEL SELECTION
    # =========================
    def select_model(self) -> nn.Module:
        torch.cuda.empty_cache()
        graph = self.dataset['graph']
        feature = self.dataset['features']

        model = DRAG(
            feature.shape[1],
            self.args.emb_size,
            gat_heads=self.args.n_head,
            num_agg_heads=self.args.n_head_agg,
            num_classes=2,
            is_concat=True,
            num_relations=len(graph.etypes),
            feat_drop=self.args.feat_drop,
            attn_drop=self.args.attn_drop
        )

        return model

    # =========================
    # PRETRAIN (EMBEDDINGS)
    # =========================
    def pretrain_embeddings(self, epochs=5):
        print("Pretraining DRAG encoder...")

        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        graph = self.dataset['graph']
        idx_train = self.dataset['idx_train']
        y_train = self.dataset['y_train']

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.args.emb_size))
        device = torch.device(f'cuda:{self.args.cuda_id}' if isinstance(self.args.cuda_id, int) else self.args.cuda_id)

        for epoch in range(epochs):
            batch_idx = generate_batch_idx(idx_train, y_train, self.args.batch_size, self.args.seed)

            train_loader = dgl.dataloading.DataLoader(
                graph,
                batch_idx,
                sampler,
                batch_size=self.args.batch_size,
                shuffle=False,
                drop_last=False,
                use_uva=True
            )

            model.train()

            for batch in train_loader:
                _, _, blocks = batch
                blocks = [b.to(device) for b in blocks]

                logits = model(blocks)
                labels = blocks[-1].dstdata['y'].long().to(device)

                loss = torch.nn.functional.cross_entropy(logits, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        print("Pretraining complete.")

    # =========================
    # TRAINING LOOP
    # =========================
    def train(self) -> Tuple[np.array, np.array]:
        self.set_seed()
        torch.cuda.empty_cache()

        cuda_id = self.args.cuda_id
        device = torch.device(f'cuda:{cuda_id}' if isinstance(cuda_id, int) else cuda_id)

        if device.type == "cuda":
            torch.cuda.set_device(device)

        graph = self.dataset['graph']
        idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']

        model = self.model
        loss_fn = nn.CrossEntropyLoss()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.args.emb_size))

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        auc_best, f1_mac_best, epoch_best = 1e-10, 1e-10, 0

        print("\n", "*" * 20, " Train the DRAG ", "*" * 20)

        for epoch in range(self.epochs):
            model.train()
            avg_loss = []
            epoch_time = 0.0
            torch.cuda.empty_cache()

            batch_idx = generate_batch_idx(idx_train, y_train, self.args.batch_size, self.args.seed)

            train_loader = dgl.dataloading.DataLoader(
                graph,
                batch_idx,
                sampler,
                batch_size=self.args.batch_size,
                shuffle=False,
                drop_last=False,
                use_uva=True
            )

            start_time = time.time()

            for batch in train_loader:
                _, output_nodes, blocks = batch
                blocks = [b.to(device) for b in blocks]

                output_labels = blocks[-1].dstdata['y'].long().to(device)

                logits = model(blocks)
                loss = loss_fn(logits, output_labels.squeeze())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                avg_loss.append(loss.item() / output_nodes.shape[0])

            epoch_time += time.time() - start_time

            line = f'Epoch: {epoch + 1} (Best: {epoch_best}), loss: {np.mean(avg_loss)}, time: {epoch_time}s'
            self.result.write_train_log(line, print_line=True)

            # =========================
            # VALIDATION
            # =========================
            if (epoch + 1) % self.args.valid_epochs == 0:
                model.eval()

                auc_val, recall_val, f1_mac_val, precision_val = test(
                    model,
                    self.dataset['valid_loader'],
                    self.result,
                    epoch,
                    epoch_best,
                    flag="val"
                )

                gain_auc = (auc_val - auc_best) / auc_best
                gain_f1_mac = (f1_mac_val - f1_mac_best) / f1_mac_best

                if (gain_auc + gain_f1_mac) > 0:
                    auc_best, recall_best, f1_mac_best, precision_best, epoch_best = \
                        auc_val, recall_val, f1_mac_val, precision_val, epoch

                    torch.save(model.state_dict(), self.result.model_path)

            # =========================
            # EARLY STOPPING
            # =========================
            if (epoch - epoch_best) > self.args.patience:
                print("\n", "*" * 20, f"Early stopping at epoch {epoch}", "*" * 20)
                break

        self.result.write_val_log(auc_best, recall_best, f1_mac_best, precision_best, epoch_best)

        print("Restore model from epoch {}".format(epoch_best))
        model.load_state_dict(torch.load(self.result.model_path))

        print("\n", "*" * 20, " Test the DRAG ", "*" * 20)

        auc_test, recall_test, f1_mac_test, precision_test = test(
            model,
            self.dataset['test_loader'],
            self.result,
            epoch,
            epoch_best,
            flag="test"
        )

        return auc_test, f1_mac_test

