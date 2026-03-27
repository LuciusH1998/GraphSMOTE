%%writefile /kaggle/working/GraphSMOTE/run.py

import argparse
import json

import torch
import dgl

from model_handler import *
from data_handler import *
from datasets import *
from utils import *
from models import *


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config_path', type=str, default='./template.json')
    args = vars(parser.parse_args())
    return args


def extract_embeddings(model_handler, data_handler):
    """
    Extract embeddings in CORRECT node order (NO batching)
    """
    print("Extracting FULL-GRAPH embeddings...")

    model = model_handler.model
    model.eval()

    graph = data_handler.dataset['graph']

    # =========================
    # FIXED DEVICE HANDLING
    # =========================
    cuda_id = model_handler.args.cuda_id
    device = torch.device(f'cuda:{cuda_id}' if isinstance(cuda_id, int) else cuda_id)

    # Move graph to device (only if CUDA)
    if device.type == "cuda":
        graph = graph.to(device)

    # Create full blocks (no sampling)
    blocks = [graph for _ in range(len(model_handler.args.emb_size))]

    with torch.no_grad():
        embeddings = model.get_embeddings(blocks)

    embeddings = embeddings.cpu()

    print("Embeddings shape:", embeddings.shape)
    print("Finished extracting embeddings.")

    return embeddings


def main(args) -> None:
    # =========================
    # STEP 1: Load config
    # =========================
    with open(args['exp_config_path']) as f:
        args = json.load(f)

    # =========================
    # STEP 2: Initialize DataHandler
    # =========================
    data_handler = DataHandlerModule(args)

    # =========================
    # STEP 3: Initialize ModelHandler
    # =========================
    model_handler = ModelHandlerModule(args, data_handler)

    # =========================
    # STEP 4: PRETRAIN (Embedding SMOTE)
    # =========================
    if args.get("use_embedding_smote", False):
        model_handler.pretrain_embeddings(epochs=5)

        # STEP 5: Extract embeddings
        embeddings = extract_embeddings(model_handler, data_handler)

        # STEP 6: Inject embeddings
        data_handler.embeddings = embeddings

        # STEP 7: Rebuild dataset with GraphSMOTE
        print("Rebuilding dataset with embedding GraphSMOTE...")
        data_handler = DataHandlerModule(args, embeddings=embeddings)

        # Rebuild model
        model_handler = ModelHandlerModule(args, data_handler)

    # =========================
    # STEP 8: Train model
    # =========================
    model_handler.train()


if __name__ == '__main__':
    args = get_arguments()
    main(args)