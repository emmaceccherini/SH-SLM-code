#%%
# we first want to finetune sbert with
# all the data but the one from one client
# and use that as test.

# Then we want to add 10% of the data from the held-out client to the training set
# then 20% etc.

# We only do this experiment on input B
# we compare SBERT 2 layers + bn and DeBERTa mean - 2 l + b.n.
#%%
import sys
import argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CLIENTS
from classification import *


#%%
MODEL_REGISTRY = {
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "deberta": "microsoft/deberta-v3-base",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run generalisability experiments for LOO clients")
    # --- the parameters that used to be hardcoded ---
    p.add_argument("--clients-train", nargs="+", required=True,
                   help="Client codes used for training/validation, e.g. C1")
    p.add_argument("--clients-test", nargs="+", required=True,
                   help="Held-out client codes, e.g. C1")
    p.add_argument("--inject-frac", type=float, default=0.0,
                   help="Fraction of held-out-client data moved into train/val (0.0-1.0)")
    p.add_argument("--random-state", "--seed", dest="random_state", type=int, default=42,
                   help="Seed for the injection sampling and the train/val split ")

    # --- other things worth exposing to the batch script ---
    p.add_argument("--models", nargs="+", default=["sbert", "deberta"],
                   choices=sorted(MODEL_REGISTRY), help="Which models to run")
    p.add_argument("--input-type", default="B")
    p.add_argument("--torch-seed", type=int, default=123,
                   help="Seed for model init / training (torch.manual_seed)")
    p.add_argument("--numpy-seed", type=int, default=42,
                   help="Seed for np.random inside run_experiment_gen")
    p.add_argument("--out-dir", type=Path, default=Path("."),
                   help="Directory for the .pkl result files")
    p.add_argument("--tag", default="",
                   help="Optional extra string appended to output filenames")

    p.add_argument("--num-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--head-lr", type=float, default=5e-5)
    p.add_argument("--layer-lr", type=float, default=1e-5)
    p.add_argument("--last-n-layers", type=int, default=2)
    p.add_argument("--min-count", type=int, default=40)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--no-finetuning", dest="finetuning", action="store_false")
    p.add_argument("--no-add-source", dest="add_source", action="store_false")
    p.set_defaults(finetuning=True, add_source=True)

    args = p.parse_args(argv)

    # fail fast on the mistakes that are easy to make in a job array
    if not 0.0 <= args.inject_frac <= 1.0:
        p.error(f"--inject-frac must be in [0, 1], got {args.inject_frac}")
    unknown = set(args.clients_train + args.clients_test) - set(CLIENTS)
    if unknown:
        p.error(f"unknown client code(s): {sorted(unknown)}; valid codes: {sorted(CLIENTS)}")
    overlap = set(args.clients_train) & set(args.clients_test)
    if overlap:
        p.error(f"--clients-train and --clients-test overlap: {sorted(overlap)}")

    return args


def create_dataset_dict_gen(input_type, return_ids = False, add_source = False,
                             val_size=0.2, min_count=40, random_state=42,
                            clients_train = None,
                            clients_test = None, inject_frac= 0.1):

    clients = clients_train + clients_test
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


    if inject_frac > 0:

        train_val_mask = mask[np.isin(all_clients, clients_train)]

        test_mask = np.isin(all_clients, clients_test)
        test_indices = np.where(test_mask)[0]

        rng = np.random.default_rng(random_state)
        test_indices = rng.permutation(test_indices)   # <-- add this

        n_inject = int(len(test_indices) * inject_frac)
        inject_indices = test_indices[:n_inject]
        test_indices = test_indices[n_inject:]
        test_mask = np.zeros_like(mask, dtype=bool)
        test_mask[test_indices] = True

        train_val_mask = np.concatenate([train_val_mask, inject_indices])
        train_val_labels = all_labels[train_val_mask]


    else:
        train_val_mask = mask[np.isin(all_clients, clients_train)]
        test_mask = np.isin(all_clients, clients_test)

        train_val_labels = all_labels[train_val_mask]


    train_mask, val_mask, _, _ = sklearn.model_selection.train_test_split(
    train_val_mask, train_val_labels,
    test_size=val_size,
    random_state=random_state, stratify=train_val_labels)

    train_idx, test_idx, val_idx = {}, {}, {}
    for c in keep_indices.keys():
        train_idx[c] = all_indices[train_mask][all_clients[train_mask] == c]
        test_idx[c]  = all_indices[test_mask][all_clients[test_mask]  == c]
        val_idx[c]   = all_indices[val_mask][all_clients[val_mask]   == c]

    train, test, val = train_idx, test_idx, val_idx

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



def prepare_dataloaders_gen(input_type, clients_train, clients_test, tokenizer, batch_size=8,
                            max_length=512, seed=42, add_source=False, min_count=40, val_size=0.2,
                            inject_frac=0.0):

    """Tokenize the dataset and build train, val, and test dataloaders."""
    datasets = create_dataset_dict_gen(input_type, return_ids = False, add_source = add_source,
                             val_size=val_size, min_count=min_count, random_state=seed,
                            clients_train = clients_train,
                            clients_test = clients_test, inject_frac=inject_frac)
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


def run_experiment_gen(train_dl, val_dl, test_dl, id2label, label2id, tokenized, checkpoint, model_factory, *,
                       input_type, device, num_epochs=30, patience=8, batch_size=8,
                   head_lr=5e-5, layer_lr=1e-6, last_n_layers=6, finetuning=True, add_source=False,
                   numpy_seed=42, torch_seed=123):
    """End-to-end run. `model_factory(checkpoint, num_labels, id2label, label2id) -> nn.Module`."""

    np.random.seed(numpy_seed); torch.manual_seed(torch_seed)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

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
    # num_warmup_steps = 0 for SBERT fine-tuning.
    # and 0.1*num_training_steps for DeBERTa fine-tuning.

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
        labels=list(range(num_labels)),
        target_names=[id2label[i] for i in range(num_labels)],
        zero_division=0,
    ))
    return {"model": model, "results": results, "logits": logits,
            "labels": labels, "label2id": label2id, "id2label": id2label}


#%%
def output_path(args, model_key):
    """Unique per-configuration filename so parallel array jobs never collide."""
    test_tag = "-".join(args.clients_test)
    train_tag = "-".join(args.clients_train)
    tag = f"_{args.tag}" if args.tag else ""
    name = (f"{model_key}_meanpool_gen_{args.input_type}"
            f"_train-{train_tag}_test-{test_tag}"
            f"_inj{args.inject_frac:g}_seed{args.random_state}{tag}.pkl")
    return args.out_dir / name


def run_one(args, model_key, device):
    checkpoint = MODEL_REGISTRY[model_key]
    print(f"\n{'='*70}\n=== {model_key} ({checkpoint})\n{'='*70}")

    train_dl, val_dl, test_dl, label2id, id2label, tokenized = prepare_dataloaders_gen(
        args.input_type,
        clients_train=args.clients_train,
        clients_test=args.clients_test,
        tokenizer=AutoTokenizer.from_pretrained(checkpoint),
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.random_state,
        add_source=args.add_source,
        min_count=args.min_count,
        val_size=args.val_size,
        inject_frac=args.inject_frac,
    )

    print("Train samples:", len(train_dl.dataset))
    print("Val samples:", len(val_dl.dataset))
    print("Test samples:", len(test_dl.dataset))

    results = run_experiment_gen(
        train_dl, val_dl, test_dl, id2label, label2id, tokenized,
        checkpoint, sbert_factory,
        input_type=args.input_type, device=device,
        num_epochs=args.num_epochs, patience=args.patience, batch_size=args.batch_size,
        head_lr=args.head_lr, layer_lr=args.layer_lr, last_n_layers=args.last_n_layers,
        finetuning=args.finetuning, add_source=args.add_source,
        numpy_seed=args.numpy_seed, torch_seed=args.torch_seed,
    )
    results["config"] = vars(args) | {"model": model_key, "checkpoint": checkpoint}

    out = output_path(args, model_key)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        torch.save(results, f)
    print(f"{model_key} results: {results['results']}  ->  {out}")
    return results


def main(argv=None):
    args = parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"input_type: {args.input_type}, inject_frac: {args.inject_frac}, "
          f"clients_train: {args.clients_train}, clients_test: {args.clients_test}, "
          f"random_state: {args.random_state}")

    for model_key in args.models:
        run_one(args, model_key, device)


if __name__ == "__main__":
    main()