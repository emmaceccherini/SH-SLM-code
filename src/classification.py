from utils import get_keep_indices
from config import input_csv, ClientsNames, CLIENTS

import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch 
import torch.nn as nn
from datasets import Dataset, DatasetDict
import copy
from datasets import Value

from sklearn.utils.class_weight import compute_class_weight

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)
from transformers.modeling_outputs import SequenceClassifierOutput



def train_test_split(input_type = "B", test_size=0.2, val_size=0.2,
                     min_count=40, random_state=23):
    """
    Positional indices into the raw input_csv rows (pre-keep_indices filtering),
    aligned with embedding order. Use them directly to subset embeddings.
    """
    keep_indices = get_keep_indices(min_count=min_count, input_type=input_type)

    all_clients, all_indices, all_labels = [], [], []
    for c, indices in keep_indices.items():
        idx_sorted = sorted(indices)
        df = pd.read_csv(input_csv(c, input_type))
        labels = df["Label"].values[idx_sorted]

        all_clients.extend([c] * len(idx_sorted))
        all_indices.extend(idx_sorted)
        all_labels.extend(labels)

    all_clients = np.array(all_clients)
    all_indices = np.array(all_indices)
    all_labels  = np.array(all_labels)

    mask = np.arange(len(all_labels))
    train_val_mask, test_mask, train_val_labels, _ = sklearn.model_selection.train_test_split(
        mask, all_labels, test_size=test_size,
        random_state=random_state, stratify=all_labels,
    )
    train_mask, val_mask, _, _ = sklearn.model_selection.train_test_split(
        train_val_mask, train_val_labels,
        test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=train_val_labels,
    )

    train_idx, test_idx, val_idx = {}, {}, {}
    for c in keep_indices.keys():
        train_idx[c] = all_indices[train_mask][all_clients[train_mask] == c]
        test_idx[c]  = all_indices[test_mask][all_clients[test_mask]  == c]
        val_idx[c]   = all_indices[val_mask][all_clients[val_mask]   == c]
    return train_idx, test_idx, val_idx

    
def create_dataset_dict(input_type, clients, return_ids = False, add_source = False, seed = 42):
    train, test, val = train_test_split(random_state=seed)
    source_map = dict(zip(CLIENTS, ClientsNames)) if add_source else None
    train_dfs, test_dfs, val_dfs = [], [], []
    for client in clients:
        df = pd.read_csv(input_csv(client, input_type))
        if add_source:
            df = df.copy()
            df["Text Input"] = source_map[client] + ". " + df["Text Input"].astype(str)
        train_dfs.append(df.iloc[train[client]])
        test_dfs.append(df.iloc[test[client]])
        val_dfs.append(df.iloc[val[client]])
    
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    val_df = pd.concat(val_dfs)


    def to_dataset(df):
        return Dataset.from_pandas(
            df[["Text Input", "Label"]].rename(columns={"Text Input": "text", "Label": "label"}),
            preserve_index=False,
        )
    
    if return_ids:
        return DatasetDict({
        "train":(train_df),
        "test": (test_df),
        "val": (val_df),
    })

    return DatasetDict({
        "train": to_dataset(train_df),
        "test": to_dataset(test_df),
        "val": to_dataset(val_df),
    })

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


class SentenceTransformerClassifier(nn.Module):
    """all-MiniLM-L6-v2 (or any SBERT model) + mean pooling + linear classifier.

    Returns a SequenceClassifierOutput so it works with the same train/eval helpers
    as AutoModelForSequenceClassification.
    """
    def __init__(self, checkpoint, num_labels, id2label=None, label2id=None):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(checkpoint)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.config = self.transformer.config
        if id2label is not None: self.config.id2label = id2label
        if label2id is not None: self.config.label2id = label2id

    @staticmethod
    def mean_pool(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, **kwargs):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        outputs = self.transformer(**kw)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        return SequenceClassifierOutput(logits=self.classifier(pooled))



def prepare_dataloaders(input_type, clients, tokenizer, batch_size=8, max_length=512, seed=42, add_source=False):
    """Tokenize the dataset and build train, val, and test dataloaders."""
    datasets = create_dataset_dict(input_type, clients, add_source=add_source, seed=seed)

    tokenized = datasets.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_length),
        batched=True,
    )
    tokenized = tokenized.remove_columns(["text"])

    unique_labels = sorted(set(datasets["train"]["label"]))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenized = tokenized.map(lambda ex: {"label": int(label2id[ex["label"]])})
    tokenized = tokenized.cast_column("label", Value("int64"))
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    g = torch.Generator()
    g.manual_seed(seed)


    train_dl = DataLoader(tokenized["train"], shuffle=True, batch_size=batch_size, collate_fn=collator)
    val_dl   = DataLoader(tokenized["val"],   batch_size=batch_size, collate_fn=collator)
    test_dl  = DataLoader(tokenized["test"],  batch_size=batch_size, collate_fn=collator)
    return train_dl, val_dl, test_dl, label2id, id2label, tokenized


def get_class_weights(tokenized_datasets, num_labels, device):
    labels_array = np.array(tokenized_datasets["train"]["labels"])
    classes = np.unique(labels_array)
    w = compute_class_weight("balanced", classes=classes, y=labels_array)
    full = np.ones(num_labels)
    for i, c in enumerate(classes):
        full[c] = w[i]
    return torch.tensor(full, dtype=torch.float).to(device)


def _base_module(model):
    """Find the base transformer module on either an HF model or our SBERT wrapper."""
    for name in ("deberta", "transformer", "bert", "roberta", "distilbert"):
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError(f"Could not locate base transformer on {type(model).__name__}")


def configure_finetuning(model, finetuning=False, last_n_layers=6):
    """Freeze the base transformer; if `finetuning`, unfreeze the last N encoder layers.

    Returns (head_params, layer_params); layer_params is empty when not finetuning.
    """
    base = _base_module(model)
    for p in base.parameters():
        p.requires_grad = False

    head_params = list(model.classifier.parameters())
    # DeBERTa exposes a separate top-level pooler; SBERT wrapper does not.
    if hasattr(model, "pooler") and not isinstance(model, SentenceTransformerClassifier):
        head_params += list(model.pooler.parameters())

    layer_params = []
    if finetuning:
        for layer in base.encoder.layer[-last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
            layer_params.extend(layer.parameters())
        n = sum(p.numel() for p in layer_params)
        print(f"Unfroze last {last_n_layers} encoder layers ({n:,} params)")
    return head_params, layer_params


def build_optimizer(model, finetuning=False, head_lr=5e-5, layer_lr=5e-5,
                    weight_decay=0.01, last_n_layers=6):
    head_params, layer_params = configure_finetuning(model, finetuning, last_n_layers)
    if finetuning:
        return AdamW(
            [{"params": head_params, "lr": head_lr},
             {"params": layer_params, "lr": layer_lr}],
            eps=1e-6, weight_decay=weight_decay,
        )
    return AdamW(head_params, lr=head_lr, eps=1e-6, weight_decay=weight_decay)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        all_logits.append(outputs.logits.cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    return compute_metrics((logits, labels)), logits, labels


def train_with_early_stopping(model, train_dl, eval_dl, optimizer, lr_scheduler,
                              loss_fn, device, num_epochs=30, patience=8):
    best_f1, best_state, no_improve = -1.0, None, 0
    progress = tqdm(range(num_epochs * len(train_dl)))

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            if torch.isnan(outputs.logits).any():
                print("NaNs in logits"); break
            loss = loss_fn(outputs.logits.float(), batch["labels"])
            loss.backward()
            optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
            progress.update(1)
            if progress.n % 100 == 0:
                print(f"Epoch {epoch+1}, Step {progress.n}, Loss: {loss.item():.4f}")

        results, _, _ = evaluate_model(model, eval_dl, device)
        print(f"Epoch {epoch+1} [val]: {results}")

        if results["weighted_f1"] > best_f1:
            best_f1 = results["weighted_f1"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            print(f"  -> New best val weighted F1: {best_f1:.4f}")
        else:
            no_improve += 1
            print(f"  -> No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    progress.close()
    return best_state, best_f1


def run_experiment(checkpoint, model_factory, *, input_type, clients, device,
                   finetuning=False, num_epochs=30, patience=8, batch_size=8,
                   head_lr=5e-5, layer_lr=1e-6, last_n_layers=6, add_source=False, random_state=42):
    """End-to-end run. `model_factory(checkpoint, num_labels, id2label, label2id) -> nn.Module`."""
    np.random.seed(42); torch.manual_seed(123)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_dl, val_dl, test_dl, label2id, id2label, tokenized = prepare_dataloaders(
        input_type, clients, tokenizer, batch_size=batch_size, add_source=add_source, seed=random_state
    )

    num_labels = len(label2id)
    model = model_factory(checkpoint, num_labels, id2label, label2id).to(device)

    class_weights = get_class_weights(tokenized, num_labels, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    # loss_fn = nn.CrossEntropyLoss()
    # class_weights = get_class_weights(tokenized, num_labels, device)
    # # use sqrt of weights to avoid overcompensating for rare classes; this is a common heuristicq
    # loss_fn = nn.CrossEntropyLoss(weight=torch.sqrt(class_weights))
    # print the parameters that we are going to use for finetuning
    print(f"Finetuning: {finetuning}, head_lr: {head_lr}, layer_lr: {layer_lr}, last_n_layers: {last_n_layers}")
    optimizer = build_optimizer(
        model, finetuning=finetuning,
        head_lr=head_lr, layer_lr=layer_lr, last_n_layers=last_n_layers,
    )
    num_training_steps = num_epochs * len(train_dl)
    print(f"num_training_steps={num_training_steps}")
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=int(0.1* num_training_steps), num_training_steps=num_training_steps,
    )


    best_state, _ = train_with_early_stopping(
        model, train_dl, val_dl, optimizer, lr_scheduler, loss_fn, device,
        num_epochs=num_epochs, patience=patience,
    )
    model.load_state_dict(best_state)

    results, logits, labels = evaluate_model(model, test_dl, device)

    if "Purchases" in label2id:
        p_id = label2id["Purchases"]
        mask = labels != p_id
        results["accuracy_excl_purchases"] = accuracy_score(
            labels[mask], np.argmax(logits[mask], axis=-1)
        )
    print("Final test results:", results)
    print("Classification report (test):")
    print(classification_report(
        labels, np.argmax(logits, axis=-1),
        target_names=[id2label[i] for i in range(num_labels)],
    ))
    return {"model": model, "results": results, "logits": logits,
            "labels": labels, "label2id": label2id, "id2label": id2label}

def deberta_factory(checkpoint, num_labels, id2label, label2id):
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id,
    )

def sbert_factory(checkpoint, num_labels, id2label, label2id):
    return SentenceTransformerClassifier(
        checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id,
    )

def split_label_counts(train_idx, val_idx, test_idx, input_type = "B"):
    """
    Counts per Label across train/val/test, aggregated over clients.
    Returns a DataFrame: rows = labels, columns = train, val, test, total.
    """
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    collected = {name: [] for name in splits}

    for c in train_idx.keys():
        df = pd.read_csv(input_csv(c, input_type))
        for name, split in splits.items():
            collected[name].extend(df["Label"].values[split[c]])

    counts = pd.DataFrame(
        {name: pd.Series(labels).value_counts() for name, labels in collected.items()}
    ).fillna(0).astype(int)
    counts["total"] = counts.sum(axis=1)
    return counts.sort_values("total", ascending=False)

class FirstTokenClassifier(nn.Module):
    """Backbone + first-token ([CLS]) pooling + single linear classifier.

    Head architecture matches SentenceTransformerClassifier exactly;
    only the pooling differs (first token vs. mean).
    """
    def __init__(self, checkpoint, num_labels, id2label=None, label2id=None):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(checkpoint)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.config = self.transformer.config
        if id2label is not None: self.config.id2label = id2label
        if label2id is not None: self.config.label2id = label2id

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, **kwargs):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        outputs = self.transformer(**kw)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = pooled.to(self.classifier.weight.dtype)   # <-- add this
        return SequenceClassifierOutput(logits=self.classifier(pooled))


def deberta_cls_factory(checkpoint, num_labels, id2label, label2id):
    return FirstTokenClassifier(
        checkpoint, num_labels=num_labels,
        id2label=id2label, label2id=label2id,
    )