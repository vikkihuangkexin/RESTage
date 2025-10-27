#!/usr/bin/env python3
"""
RESTage training and cross-validation script.
"""
import os
import datetime
import subprocess
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
import torch
from transformers import BertConfig, Trainer
from transformers.training_args import TrainingArguments
from tqdm.notebook import tqdm
from geneformer import DataCollatorForGeneClassification
from geneformer.pretrainer import token_dictionary
from model import RESTage

# Configure visible GPU(s)
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"

# Server code string used in output paths
servers = "demo"

def main(organ, cl_num):
    # Read gene metadata and create mapping from gene name to Ensembl id
    gene_info = pd.read_csv("data_demo/gene_info_table.csv", index_col=0)
    gene_name_id_dict = dict(zip(gene_info["gene_name"], gene_info["ensembl_id"]))

    # Task type: 'class' or 'regression'
    tasks = "class"

    # Result directory
    result_dir = f"check/{organ}"
    os.makedirs(result_dir, exist_ok=True)

    # Whether to reinitialize embeddings (kept for compatibility)
    reset_embedding = False

    # Prepare target token ids and labels from two gene groups
    def prep_inputs(genegroup1, genegroup2, id_type):
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

        # Positive class for group2, negative for group1
        targets = np.array(targets2_id + targets1_id)
        labels = np.array([1] * len(targets2_id) + [0] * len(targets1_id))

        nsplits = min(5, min(len(targets2_id), len(targets1_id)) - 1)
        assert nsplits > 2, "Not enough examples to create at least 3 splits"

        print(f"# targets positive: {len(targets2_id)}\n# targets negative: {len(targets1_id)}\n# splits: {nsplits}")
        return targets, labels, nsplits

    # Load label file for the organ and prepare positive/negative gene sets
    dosage_tfs = pd.read_csv(f"data_demo/senescence/{organ}/Training/label_ensembl.csv", header=0)
    sensitive = dosage_tfs["SASP_interaction"].dropna()
    insensitive = dosage_tfs["non_interaction"].dropna()

    targets, labels, nsplits = prep_inputs(insensitive, sensitive, "ensembl_id")

    # Load dataset saved with HuggingFace datasets
    train_dataset = load_from_disk(f"data_demo/senescence/{organ}/Training/{organ}.dataset")
    shuffled_train_dataset = train_dataset.shuffle(seed=42)
    subsampled_train_dataset = shuffled_train_dataset

    # Padding and formatting helper for batches before model forward pass
    def preprocess_classifier_batch(cell_batch, max_len):
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

        padded_batch = cell_batch.map(pad_label_example)
        return padded_batch

    # Softmax helper for converting logits to probabilities
    def py_softmax(vector):
        e = np.exp(vector)
        return e / e.sum()

    # Simple vote helper to map token logits to predicted class
    def vote(logit_pair):
        a, b = logit_pair
        if a > b:
            return 0
        elif b > a:
            return 1
        else:
            return "tie"

    # Plotting and metric helpers
    def auc_plot(predict_logits, predict_labels, ksplit_output_dir, label_type, mean_fpr):
        logits_by_cell = torch.cat(predict_logits)
        all_logits = logits_by_cell.reshape(-1, logits_by_cell.shape[2])
        labels_by_cell = torch.cat(predict_labels)
        all_labels = torch.flatten(labels_by_cell)
        paired = [item for item in list(zip(all_logits.tolist(), all_labels.tolist())) if item[1] != -100]
        y_pred = [vote(item[0]) for item in paired]
        y_true = [item[1] for item in paired]
        logits_list = [item[0] for item in paired]
        y_score = [py_softmax(item)[1] for item in logits_list]
        conf_mat = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.savefig(os.path.join(ksplit_output_dir, f"auc_{label_type}.png"))
        plt.savefig(os.path.join(ksplit_output_dir, f"auc_{label_type}.pdf"))
        plt.close()

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        return [fpr, tpr, interp_tpr, conf_mat]

    def auc_plot_C(predict_logits, predict_labels, ksplit_output_dir, label_type, mean_fpr):
        # Same as auc_plot but handles different logits shape for cell-level predictions
        all_logits = torch.cat(predict_logits)
        labels_by_cell = torch.cat(predict_labels)
        all_labels = torch.flatten(labels_by_cell)
        paired = [item for item in list(zip(all_logits.tolist(), all_labels.tolist())) if item[1] != -100]
        y_pred = [vote(item[0]) for item in paired]
        y_true = [item[1] for item in paired]
        logits_list = [item[0] for item in paired]
        y_score = [py_softmax(item)[1] for item in logits_list]
        conf_mat = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.savefig(os.path.join(ksplit_output_dir, f"auc_{label_type}.png"))
        plt.savefig(os.path.join(ksplit_output_dir, f"auc_{label_type}.pdf"))
        plt.close()

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        return [fpr, tpr, interp_tpr, conf_mat]

    # Helper: largest integer <= N divisible by K
    def find_largest_div(N, K):
        rem = N % K
        return N if (rem == 0) else N - rem

    # Evaluate model on an evaluation dataset in batches and compute metrics
    def classifier_predict_multi(model, evalset, forward_batch_size, mean_fpr, ksplit_output_dir):
        predict_logits_C = []
        predict_labels_C = []
        predict_logits_G = []
        predict_labels_G = []
        model.eval()

        evalset_len = len(evalset)
        max_divisible = find_largest_div(evalset_len, forward_batch_size)
        if len(evalset) - max_divisible == 1:
            evalset_len = max_divisible

        max_evalset_len = max(evalset.select([i for i in range(evalset_len)])["length"])

        for i in range(0, evalset_len, forward_batch_size):
            max_range = min(i + forward_batch_size, evalset_len)
            batch_evalset = evalset.select([j for j in range(i, max_range)])
            padded_batch = preprocess_classifier_batch(batch_evalset, max_evalset_len)
            padded_batch.set_format(type="torch")

            input_data_batch = padded_batch["input_ids"]
            attn_msk_batch = padded_batch["attention_mask"]
            label_batch = padded_batch["labels"]

            with torch.no_grad():
                outputs = model(
                    input_ids=input_data_batch.to("cuda"),
                    attention_mask=attn_msk_batch.to("cuda"),
                    labels=label_batch.to("cuda"),
                )
                predict_logits_G += [torch.squeeze(outputs["token"].logits.to("cpu"))]
                predict_labels_G += [torch.squeeze(label_batch[:, 0, :].to("cpu"))]
                predict_logits_C += [torch.squeeze(outputs["cls"].logits.to("cpu"))]
                predict_labels_C += [torch.squeeze(label_batch[:, 1][:, 0].to("cpu"))]

        cellauc = auc_plot_C(predict_logits_C, predict_labels_C, ksplit_output_dir, "cell", mean_fpr)
        geneauc = auc_plot(predict_logits_G, predict_labels_G, ksplit_output_dir, "gene", mean_fpr)
        return cellauc, geneauc

    # Compute weighted mean TPR and AUC standard deviation across folds
    def get_cross_valid_metrics(all_tpr, all_roc_auc, all_tpr_wt):
        wts = [count / sum(all_tpr_wt) for count in all_tpr_wt]
        all_weighted_tpr = [a * b for a, b in zip(all_tpr, wts)]
        mean_tpr = np.sum(all_weighted_tpr, axis=0)
        mean_tpr[-1] = 1.0
        all_weighted_roc_auc = [a * b for a, b in zip(all_roc_auc, wts)]
        roc_auc = np.sum(all_weighted_roc_auc)
        roc_auc_sd = math.sqrt(np.average((all_roc_auc - roc_auc) ** 2, weights=wts))
        return mean_tpr, roc_auc, roc_auc_sd

    # Cross-validation training and evaluation
    def cross_validate(data, targets, labels, nsplits, subsample_size, training_args, freeze_layers, output_dir, num_proc):
        model_dir_test = os.path.join(output_dir, "ksplit0/models/pytorch_model.bin")
        if os.path.isfile(model_dir_test):
            raise Exception("Model already saved to this directory.")

        num_classes = len(set(labels))
        mean_fpr = np.linspace(0, 1, 100)

        all_tpr_C, all_roc_auc_C, all_tpr_wt_C = [], [], []
        confusion_C = np.zeros((num_classes, num_classes))

        all_tpr_G, all_roc_auc_G, all_tpr_wt_G = [], [], []
        confusion_G = np.zeros((num_classes, num_classes))
        label_dicts = []

        skf = StratifiedKFold(n_splits=nsplits, random_state=0, shuffle=True)
        iteration_num = 0

        for train_index, eval_index in tqdm(skf.split(targets, labels)):
            # Ensure a set of control tokens always in train_index (original logic kept)
            train_index = sorted(set(train_index) | set(range(0, cl_num)))
            eval_index = [i for i in eval_index if i > cl_num]

            if len(labels) > 500:
                nsplits = 3
                if iteration_num == 3:
                    break

            targets_train, targets_eval = targets[train_index], targets[eval_index]
            labels_train, labels_eval = labels[train_index], labels[eval_index]
            label_dict_train = dict(zip(targets_train, labels_train))
            label_dict_eval = dict(zip(targets_eval, labels_eval))
            label_dicts.append((iteration_num, targets_train, targets_eval, labels_train, labels_eval))

            # Filter functions for dataset examples containing train/eval tokens
            def if_contains_train_label(example):
                return not set(label_dict_train.keys()).isdisjoint(example["input_ids"])

            def if_contains_eval_label(example):
                return not set(label_dict_eval.keys()).isdisjoint(example["input_ids"])

            # Filter datasets for this fold
            trainset = data.filter(if_contains_train_label, num_proc=num_proc)
            evalset = data.filter(if_contains_eval_label, num_proc=num_proc)

            # Subsample and split evaluation set into in-fold and OOS evaluation
            training_size = min(subsample_size, int(len(trainset) * 0.6))
            trainset_min = trainset.select([i for i in range(training_size)])
            eval_size = len(evalset) - training_size
            half_training_size = round(eval_size / 2)
            evalset_train_min = evalset.select([i for i in range(training_size, training_size + half_training_size)])
            evalset_oos_min = evalset.select([i for i in range(training_size + half_training_size, training_size + eval_size)])

            # Label generation functions (classification)
            if tasks != "regression":

                def generate_train_labels(example):
                    label_cell = 1 if example["group"] == "old" else 0
                    example["labels"] = (
                        [label_dict_train.get(token_id, -100) for token_id in example["input_ids"]],
                        [label_cell for _ in example["input_ids"]],
                    )
                    return example

                def generate_eval_labels(example):
                    label_cell = 1 if example["group"] == "old" else 0
                    example["labels"] = (
                        [label_dict_eval.get(token_id, -100) for token_id in example["input_ids"]],
                        [label_cell for _ in example["input_ids"]],
                    )
                    return example
            else:
                # Regression mode kept for completeness
                def generate_train_labels(example):
                    label_cell = example["age"]
                    example["labels"] = (
                        [label_dict_train.get(token_id, -100) for token_id in example["input_ids"]],
                        [label_cell for _ in example["input_ids"]],
                    )
                    return example

                def generate_eval_labels(example):
                    label_cell = example["age"]
                    example["labels"] = (
                        [label_dict_eval.get(token_id, -100) for token_id in example["input_ids"]],
                        [label_cell for _ in example["input_ids"]],
                    )
                    return example

            # Apply label mapping to datasets
            trainset_labeled = trainset_min.map(generate_train_labels)
            evalset_train_labeled = evalset_train_min.map(generate_eval_labels)
            evalset_oos_labeled = evalset_oos_min.map(generate_eval_labels)

            # Create per-split output directories
            ksplit_output_dir = os.path.join(output_dir, f"ksplit{iteration_num}")
            ksplit_model_dir = os.path.join(ksplit_output_dir, "models/")

            model_output_file = os.path.join(ksplit_model_dir, "pytorch_model.bin")
            if os.path.isfile(model_output_file):
                raise Exception("Model already saved to this directory.")

            subprocess.call(f"mkdir -p {ksplit_output_dir}", shell=True)
            subprocess.call(f"mkdir -p {ksplit_model_dir}", shell=True)

            # Initialize model and optionally freeze lower layers
            model_config = BertConfig("demo_model")
            num_labels_cell = 1 if tasks == "regression" else 2
            model = RESTage(model_config, [2, num_labels_cell], modelpath=r"demo_model")

            if freeze_layers is not None:
                modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
                for module in modules_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

            if reset_embedding:
                import torch.nn as nn
                for modules in model.bert.embeddings.modules():
                    if isinstance(modules, (nn.Linear, nn.Embedding)):
                        modules.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
                    elif isinstance(modules, nn.LayerNorm):
                        modules.bias.data.zero_()
                        modules.weight.data.fill_(1.0)
                    if isinstance(modules, nn.Linear) and modules.bias is not None:
                        modules.bias.data.zero_()

            model = model.to("cuda:0")

            # Set output dir in training args and create Trainer
            training_args["output_dir"] = ksplit_output_dir
            training_args_init = TrainingArguments(**training_args)

            trainer = Trainer(
                model=model,
                args=training_args_init,
                data_collator=DataCollatorForGeneClassification(),
                train_dataset=trainset_labeled,
                eval_dataset=evalset_train_labeled,
            )

            # Train and save model for this split
            trainer.train()
            trainer.save_model(ksplit_model_dir)

            # Evaluate model on OOS evaluation data
            cells, genes = classifier_predict_multi(trainer.model, evalset_oos_labeled, geneformer_batch_size, mean_fpr, ksplit_output_dir)

            # Aggregate metrics
            confusion_G = confusion_G + genes[3]
            all_tpr_G.append(genes[2])
            all_roc_auc_G.append(auc(genes[0], genes[1]))
            all_tpr_wt_G.append(len(genes[1]))

            if tasks != "regression":
                confusion_C = confusion_C + cells[3]
                all_tpr_C.append(cells[2])
                all_roc_auc_C.append(auc(cells[0], cells[1]))
                all_tpr_wt_C.append(len(cells[1]))
            else:
                all_tpr_C.append(cells[0])
                all_tpr_wt_C.append(cells[1])

            iteration_num += 1

        # Compute cross-validated metrics and return
        if tasks != "regression":
            mean_tpr_C, roc_auc_C, roc_auc_sd_C = get_cross_valid_metrics(all_tpr_C, all_roc_auc_C, all_tpr_wt_C)
            mean_tpr_G, roc_auc_G, roc_auc_sd_G = get_cross_valid_metrics(all_tpr_G, all_roc_auc_G, all_tpr_wt_G)

            return (
                [all_roc_auc_C, roc_auc_C, roc_auc_sd_C, mean_fpr, mean_tpr_C, confusion_C, label_dicts],
                [all_roc_auc_G, roc_auc_G, roc_auc_sd_G, mean_fpr, mean_tpr_G, confusion_G, label_dicts],
            )
        else:
            mean_tpr_G, roc_auc_G, roc_auc_sd_G = get_cross_valid_metrics(all_tpr_G, all_roc_auc_G, all_tpr_wt_G)
            return [all_tpr_C, all_tpr_wt_C], [all_roc_auc_G, roc_auc_G, roc_auc_sd_G, mean_fpr, mean_tpr_G, confusion_G, label_dicts]

    # Plot ROC from aggregated fold results and save CSV with curve points
    def plot_ROC(bundled_data, title, savepath, types):
        plt.figure()
        lw = 2
        all_data = []
        for roc_auc, roc_auc_sd, mean_fpr, mean_tpr, sample, color in bundled_data:
            plt.plot(mean_fpr, mean_tpr, color=color, lw=lw, label=f"{sample} (AUC {roc_auc:0.2f} ± {roc_auc_sd:0.2f})")
            for j in range(len(mean_fpr)):
                all_data.append(
                    {
                        "curve_id": 0,
                        "sample": sample,
                        "color": color,
                        "roc_auc": roc_auc,
                        "roc_auc_sd": roc_auc_sd,
                        "fpr": mean_fpr[j],
                        "tpr": mean_tpr[j],
                        "point_index": j,
                    }
                )

        df = pd.DataFrame(all_data)
        os.makedirs(savepath, exist_ok=True)
        df.to_csv(f"{savepath}/{types}_data.csv", index=False)
        plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(savepath, f"auc_{types}.png"))
        plt.savefig(os.path.join(savepath, f"auc_{types}.pdf"))
        plt.close()

    # Model and training hyperparameters (kept as in original)
    max_input_size = 2 ** 11  # 2048
    max_lr = 5e-2
    freeze_layers = 4
    num_proc = 12
    geneformer_batch_size = 20
    lr_schedule_fn = "linear"
    warmup_steps = 500
    epochs = 10
    optimizer = "adamw"
    subsample_size = 10_000

    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "evaluation_strategy": "no",
        "save_strategy": "epoch",
        "logging_steps": 100,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
    }

    # Construct output directory with datestamp and hyperparameters
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    training_output_dir = (
        f"{result_dir}/multi_{tasks}_group_fullGene_{datestamp}_geneformer_GeneClassifier_"
        f"N1network_L{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_"
        f"WU{warmup_steps}_E{epochs}_O{optimizer}_n{subsample_size}_F{freeze_layers}_"
        f"ResetEmbed{str(reset_embedding)}_servercode-{servers}/"
    )

    ksplit_model_test = os.path.join(training_output_dir, "ksplit0/models/pytorch_model.bin")
    if os.path.isfile(ksplit_model_test):
        raise Exception("Model already saved to this directory.")

    subprocess.call(f"mkdir -p {training_output_dir}", shell=True)

    # Run cross-validation
    cells, genes = cross_validate(
        subsampled_train_dataset,
        targets,
        labels,
        nsplits,
        subsample_size,
        training_args,
        freeze_layers,
        training_output_dir,
        num_proc=1,
    )

    # Plot aggregated ROC curves and save results
    if tasks != "regression":
        bundled_data1 = [(cells[1], cells[2], cells[3], cells[4], "Geneformer", "red")]
        bundled_data2 = [(genes[1], genes[2], genes[3], genes[4], "Geneformer", "red")]
        plot_ROC(bundled_data1, "Dosage Sensitive vs Insensitive TFs", training_output_dir, "cell")
        plot_ROC(bundled_data2, "Dosage Sensitive vs Insensitive TFs", training_output_dir, "gene")
    else:
        bundled_data2 = [(genes[1], genes[2], genes[3], genes[4], "Geneformer", "red")]
        plot_ROC(bundled_data2, "Dosage Sensitive vs Insensitive TFs", training_output_dir, "gene")


if __name__ == "__main__":
    organ = "skin"
    data_info = pd.read_csv(f"data_demo/senescence/{organ}/data_information.csv")
    cl_num = int(data_info["cy_num"].tolist()[0])
    main(organ, cl_num)
