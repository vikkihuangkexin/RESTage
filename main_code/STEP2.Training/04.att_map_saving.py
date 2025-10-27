#!/usr/bin/env python3
"""
Generate attention maps and related outputs from a pretrained RESTage model.

"""

import os
import subprocess
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch
from transformers import BertConfig
from tqdm.notebook import tqdm
from collections import defaultdict
from geneformer import DataCollatorForGeneClassification
from geneformer.pretrainer import token_dictionary
from model import RESTage

# Configure visible GPU(s)
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"

# ---------- Helper functions ----------

def prep_inputs(genegroup1, genegroup2, id_type, gene_name_id_dict):
    """
    Convert two gene groups (by gene_name or ensembl_id) to token ids and labels.
    Returns (targets_array, labels_array, nsplits)
    """
    if id_type == "gene_name":
        targets1 = [gene_name_id_dict[gene] for gene in genegroup1 if gene_name_id_dict.get(gene) in token_dictionary]
        targets2 = [gene_name_id_dict[gene] for gene in genegroup2 if gene_name_id_dict.get(gene) in token_dictionary]
    elif id_type == "ensembl_id":
        targets1 = [gene for gene in genegroup1 if gene in token_dictionary]
        targets2 = [gene for gene in genegroup2 if gene in token_dictionary]
    else:
        raise ValueError("id_type must be 'gene_name' or 'ensembl_id'")

    targets1_id = [token_dictionary[gene] for gene in targets1]
    targets2_id = [token_dictionary[gene] for gene in targets2]

    targets = np.array(targets2_id + targets1_id)
    labels = np.array([1] * len(targets2_id) + [0] * len(targets1_id))

    nsplits = min(5, min(len(targets2_id), len(targets1_id)) - 1)
    assert nsplits > 2, "Not enough examples to create at least 3 splits"

    print(f"# positive targets: {len(targets2_id)}\n# negative targets: {len(targets1_id)}\n# splits: {nsplits}")
    return targets, labels, nsplits


def preprocess_classifier_batch(cell_batch, max_len):
    """
    Pad input_ids, labels and generate attention_mask for a HF Dataset batch.
    """
    if max_len is None:
        max_len = max([len(i) for i in cell_batch["input_ids"]])

    def pad_label_example(example):
        example["labels"] = np.pad(
            example["labels"],
            (0, max_len - len(example["input_ids"])),
            mode="constant",
            constant_values=-100,
        )
        example["input_ids"] = np.pad(
            example["input_ids"],
            (0, max_len - len(example["input_ids"])),
            mode="constant",
            constant_values=token_dictionary.get("<pad>"),
        )
        example["attention_mask"] = (example["input_ids"] != token_dictionary.get("<pad>")).astype(int)
        return example

    return cell_batch.map(pad_label_example)


def find_largest_div(N, K):
    """
    Return the largest integer <= N that is divisible by K.
    """
    rem = N % K
    return N if rem == 0 else N - rem


def find_keys_by_value_optimized_defaultdict(d, values):
    """
    Reverse lookup optimized with a defaultdict prepared on the fly.
    For a list of values, return corresponding keys (first match) or the value itself if not found.
    """
    reverse_dict = defaultdict(list)
    for k, v in d.items():
        reverse_dict[v].append(k)
    return [reverse_dict[v][0] if v in reverse_dict else v for v in values]


# ---------- Attention-map extraction and saving ----------

def save_mask(data, idx, gene_names, out_dir, layer_name):
    """
    Save a heatmap image and CSV of the provided matrix `data`.
    - data: 2D numpy array
    - gene_names: list-like column/row names
    - out_dir: directory to save files
    - layer_name: string to include in filenames
    """
    os.makedirs(out_dir, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    plt.pcolor(data, cmap=cmap)
    plt.colorbar()
    plt.title("Attention map")
    img_path = os.path.join(out_dir, f"mask_{layer_name}_{idx}.png")
    plt.savefig(img_path)
    plt.close()

    df = pd.DataFrame(data, index=gene_names, columns=gene_names)
    csv_path = os.path.join(out_dir, f"mask_{layer_name}_{idx}.csv")
    df.to_csv(csv_path)


def classifier_predict(model, evalset, forward_batch_size, output_dir, pos_S, gene_id_dict, cy_id, cy_id_dict, cy_inputsID_dict):
    """
    Run model in inference mode over `evalset`, extract attention maps (mean of heads),
    and save per-example attention matrices for cytokines of interest.
    - model: a PyTorch model (can be DataParallel)
    - evalset: HF Dataset
    - forward_batch_size: batch size for inference
    - output_dir: directory where outputs will be written
    - pos_S: ordered list of gene symbols used as reference column order
    - gene_id_dict: mapping from Ensembl -> gene symbol (or similar)
    - cy_id: list of token ids corresponding to cytokines of interest
    - cy_id_dict: mapping from token id -> gene symbol for cytokines
    - cy_inputsID_dict: mapping used to convert inputs to cytokine ids (if relevant)
    """
    model.eval()
    evalset_len = len(evalset)
    max_divisible = find_largest_div(evalset_len, forward_batch_size)
    if len(evalset) - max_divisible == 1:
        evalset_len = max_divisible

    max_eval_len = max(evalset.select([i for i in range(evalset_len)])["length"])

    rows = []
    # header with gene list
    header_df = pd.DataFrame({"gene_name": pos_S})
    rows.append(header_df)

    for start in tqdm(range(0, evalset_len, forward_batch_size)):
        max_range = min(start + forward_batch_size, evalset_len)
        batch_evalset = evalset.select([i for i in range(start, max_range)])
        padded_batch = preprocess_classifier_batch(batch_evalset, max_eval_len)
        padded_batch.set_format(type="torch")

        input_ids = padded_batch["input_ids"]
        attn_mask = padded_batch["attention_mask"]
        label_batch = padded_batch["labels"]
        IDs = padded_batch.get("ID", [f"idx_{i}" for i in range(len(input_ids))])
        groups = padded_batch.get("group", ["NA"] * len(input_ids))
        cell_types = padded_batch.get("cell_type", ["NA"] * len(input_ids))
        barcodes = padded_batch.get("barcode", [f"bc_{i}" for i in range(len(input_ids))])

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to("cuda"),
                attention_mask=attn_mask.to("cuda"),
                labels=label_batch.to("cuda"),
                output_attentions=True,
            )

            # outputs['cls'].attentions expected as tuple(layers) each with shape (batch, heads, seq_len, seq_len)
            # Compute mean across heads to get (layers, batch, seq_len, seq_len) -> take layer index, mean heads
            attentions = outputs["cls"].attentions  # keep using 'cls' key as in original code
            attn_stack = torch.stack(attentions)  # (num_layers, batch, heads, seq_len, seq_len)
            # mean across heads dimension -> (num_layers, batch, seq_len, seq_len)
            attn_mean_heads = attn_stack.mean(dim=2).cpu().numpy()

            # Example: take layer 6 (index 5) as in original script
            layer_idx = 5
            if layer_idx >= attn_mean_heads.shape[0]:
                layer_idx = 0
            layer_matrix = attn_mean_heads[layer_idx]  # shape (batch, seq_len, seq_len)

            for idx_in_batch in range(layer_matrix.shape[0]):
                seq_matrix = layer_matrix[idx_in_batch]  # (seq_len, seq_len)
                input_seq = list(input_ids[idx_in_batch].cpu().numpy())
                # Map token ids back to ensembl ids then to gene symbols
                ensmbl_list = find_keys_by_value_optimized_defaultdict(token_dictionary, input_seq)
                symbol_list = find_keys_by_value_optimized_defaultdict(gene_id_dict, ensmbl_list)

                # Ensure we sort and align to symbol_list
                maps_df = pd.DataFrame(seq_matrix, index=symbol_list, columns=symbol_list)
                maps_df.sort_index(axis=0, inplace=True)
                maps_df.sort_index(axis=1, inplace=True)

                # Identify cytokine columns that are in the input sequence
                cytokine_tokens = [tok for tok in cy_id if tok in input_seq]
                cytokine_symbols = []
                for tok in cytokine_tokens:
                    # convert token id to symbol if mapping exists; fallback to token id
                    cytokine_symbols.append(cy_id_dict.get(tok, tok))

                # For each cytokine symbol, extract column values ordered by pos_S and append to final rows
                for cy_sym in cytokine_symbols:
                    col_values = [maps_df.get(cy_sym, {}).get(g, 0) if cy_sym in maps_df.columns else 0 for g in pos_S]
                    col_name = f"{cy_sym}-{barcodes[idx_in_batch]}-{IDs[idx_in_batch]}-{groups[idx_in_batch]}-{cell_types[idx_in_batch]}"
                    rows.append(pd.DataFrame({col_name: col_values}))

    # Concatenate and save
    final_df = pd.concat(rows, axis=1)
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, "cytokine_fullmap.csv"))
    print("Predict end")


# ---------- Minimal cross-validation helpers (kept for compatibility) ----------

def get_cross_valid_metrics(all_tpr, all_roc_auc, all_tpr_wt):
    """
    Weighted mean TPR and AUC SD across folds.
    """
    wts = [count / sum(all_tpr_wt) for count in all_tpr_wt]
    all_weighted_tpr = [a * b for a, b in zip(all_tpr, wts)]
    mean_tpr = np.sum(all_weighted_tpr, axis=0)
    mean_tpr[-1] = 1.0
    all_weighted_roc_auc = [a * b for a, b in zip(all_roc_auc, wts)]
    roc_auc = np.sum(all_weighted_roc_auc)
    roc_auc_sd = math.sqrt(np.average((np.array(all_roc_auc) - roc_auc) ** 2, weights=wts))
    return mean_tpr, roc_auc, roc_auc_sd


# ---------- Generate function to load model and call prediction ----------

def Generate(data, targets, labels, nsplits, subsample_size, training_args, freeze_layers, model_dir, out_dir, helpers):
    """
    Load a model from `model_dir/models` and run classifier_predict on OOS evaluation set.
    - helpers: dict with required auxiliary mappings:
        { 'pos_S', 'gene_id_dict', 'cy_id', 'cy_id_dict', 'cy_inputsID_dict' }
    """
    model_dir_models = os.path.join(model_dir, "models")
    if not os.path.isdir(model_dir_models):
        raise FileNotFoundError(f"{model_dir_models} not found")

    # minimal splitting to pick evalset for attention extraction (keeps original logic)
    skf = StratifiedKFold(n_splits=nsplits, random_state=0, shuffle=True)
    for train_index, eval_index in skf.split(targets, labels):
        # filter dataset entries that contain any eval tokens
        targets_eval = targets[eval_index]
        label_dict_eval = dict(zip(targets_eval, np.array(labels)[eval_index]))

        def if_contains_eval_label(example):
            return not set(label_dict_eval.keys()).isdisjoint(example["input_ids"])

        evalset = data.filter(if_contains_eval_label, num_proc=1)
        # We use the full evalset for attention extraction; break after first fold
        break

    # prepare output path for attention maps
    att_map_dir = os.path.join(out_dir, "att_map")
    os.makedirs(att_map_dir, exist_ok=True)

    # load model config and model weights using RESTage
    model_config = BertConfig(model_dir)
    num_labels_cell = 2  # classification mode; adjust if regression needed
    model = RESTage(model_config, [2, num_labels_cell], modelpath=model_dir)
    # move to CUDA and DataParallel if multiple GPUs are visible
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to("cuda")

    # run prediction and save cytokine maps
    classifier_predict(
        model=model,
        evalset=evalset,
        forward_batch_size=20,
        output_dir=att_map_dir,
        pos_S=helpers["pos_S"],
        gene_id_dict=helpers["gene_id_dict"],
        cy_id=helpers["cy_id"],
        cy_id_dict=helpers["cy_id_dict"],
        cy_inputsID_dict=helpers.get("cy_inputsID_dict", {}),
    )


# ---------- Main entry point ----------

def main(organ, cl_num):
    """
    Example main showing how to configure file paths and call Generate.
    Users should adapt paths to their environment before running.
    """
    # Load gene metadata
    gene_info = pd.read_csv("data_demo/gene_info_table.csv", index_col=0)
    gene_name_id_dict = dict(zip(gene_info["gene_name"], gene_info["ensembl_id"]))

    # Load label files (paths are placeholders; change to your repo layout)
    dosage_tfs = pd.read_csv(f"data_demo/senescence/{organ}/Training/label_ensembl.csv", header=0)
    dosage_pos = pd.read_csv(f"data_demo/senescence/{organ}/Training/pos_P.csv", header=0)
    dosage_neg = pd.read_csv(f"data_demo/senescence/{organ}/Training/neg_P.csv", header=0)

    pos_S = list(dosage_pos["SYMBOL"])
    pos_E = list(dosage_pos["ENSEMBL"])

    # Prepare cytokine token ids using token_dictionary
    cy_num = cl_num
    cy_id = [token_dictionary[gene] for gene in pos_E[:cy_num] if gene in token_dictionary]
    cy_em = [gene for gene in pos_E[:cy_num] if gene in token_dictionary]

    # simple gene id -> symbol mapping (example)
    gene_id_dict = {ens: sym for ens, sym in zip(pos_E, pos_S)}

    sensitive = dosage_tfs["SASP_interaction"].dropna()
    insensitive = dosage_tfs["non_interaction"].dropna()
    targets, labels, nsplits = prep_inputs(insensitive, sensitive, "ensembl_id", gene_name_id_dict)

    # Load dataset (path is a placeholder)
    train_dataset = load_from_disk(f"data_demo/senescence/{organ}/{organ}_fullGene.dataset")
    shuffled = train_dataset.shuffle(seed=42)

    # Training args placeholders (kept to match original function signature)
    training_args = {
        "learning_rate": 5e-2,
        "do_train": True,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "num_train_epochs": 1,
    }

    # Output dir for attention maps and results
    out_dir = f"outputs/{organ}/"
    os.makedirs(out_dir, exist_ok=True)

    # Helper dict passed to Generate
    helpers = {
        "pos_S": pos_S,
        "gene_id_dict": gene_id_dict,
        "cy_id": cy_id,
        "cy_id_dict": {token_dictionary[g]: g for g in cy_em},
        "cy_inputsID_dict": {},  # keep empty or fill if needed
    }

    # Call Generate to produce attention maps
    Generate(
        data=shuffled,
        targets=targets,
        labels=labels,
        nsplits=nsplits,
        subsample_size=10000,
        training_args=training_args,
        freeze_layers=4,
        model_dir="demo_model",  # change to your pretrained model dir
        out_dir=out_dir,
        helpers=helpers,
    )


if __name__ == "__main__":
    organ = "skin"
    data_info = pd.read_csv(f"data_demo/senescence/{organ}/data_information.csv")
    cl_num = int(data_info["cy_num"].tolist()[0])
    main(organ, cl_num)
