"""Real held-out evaluation for torchvision's ImageNet classifier."""

from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from pyarrow.parquet import read_metadata
from torch.nn import functional as F
from torch.utils.data import Dataset, IterableDataset

DATASET = "ILSVRC/imagenet-1k"


def label_mapping(names, categories):
    """Validate the official sorted-synset ImageNet label order.

    HF uses comma-separated synonyms; torchvision uses shorter display names.
    The official dataset and weights both use the 1000 sorted synsets.
    Check all labels against their synonyms instead of comparing display strings.
    """
    if len(names) != 1000 or len(categories) != 1000:
        raise ValueError("Expected the original 1000 ImageNet classes")
    # These three display names differ in the official HF metadata.
    # Keep the bird and machine 'crane' classes distinct by checking their IDs.
    aliases = {
        134: ("crane", "crane bird"),
        517: ("crane2", "crane"),
        639: ("maillot, tank suit", "maillot tank suit"),
    }
    for index, (name, category) in enumerate(zip(names, categories, strict=True)):
        if (name, category) == aliases.get(index):
            continue
        if category.casefold() not in {part.strip().casefold() for part in name.split(",")}:
            raise ValueError(f"Dataset class order does not match pretrained category {index}")
    return tuple(range(1000))


class Images(Dataset):
    """Lazy RGB decoding and the exact preprocessing associated with the weights."""

    def __init__(self, rows, transform, mapping):
        self.rows, self.transform, self.mapping = rows, transform, mapping

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[int(index)]
        return self.transform(row["image"].convert("RGB")), self.mapping[row["label"]]


class TrainingImages(IterableDataset):
    """Stream the requested training samples from local Parquet shards."""

    def __init__(self, rows, transform, mapping, size):
        self.rows, self.transform, self.mapping, self.size = rows, transform, mapping, size

    def __len__(self):
        return self.size

    def __iter__(self):
        for row in self.rows:
            yield self.transform(row["image"].convert("RGB")), self.mapping[row["label"]]


def load_images(
    weights, data_dir=None, *, need_train=False, train_samples=512, val_samples=0, seed=7
):
    """Read HF ImageNet Parquet shards downloaded with hf download.

    Resolve the default HF cache offline unless a directory is explicitly supplied.
    Only validation is required by default. Training uses a separate local split;
    limits of zero select the full split. This function never downloads data.
    """
    if train_samples < 0 or val_samples < 0:
        raise ValueError("Sample limits must be nonnegative")
    splits = ("validation", "train") if need_train else ("validation",)
    root = Path(
        data_dir
        if data_dir is not None
        else snapshot_download(
            DATASET,
            repo_type="dataset",
            allow_patterns=[f"data/{split}-*.parquet" for split in splits],
            local_files_only=True,
        )
    ).resolve()
    files = {}
    for split in splits:
        files[split] = [str(path) for path in sorted((root / "data").glob(f"{split}-*.parquet"))]
        if not files[split]:
            raise FileNotFoundError(f"Missing ImageNet {split} shards under {root / 'data'}")
    metadata = {"dataset": DATASET, "data_dir": str(root), "files": files, "subset_seed": seed}
    rows = load_dataset(
        "parquet",
        split="validation",
        data_files={"validation": files["validation"]},
    )
    mapping = label_mapping(rows.features["label"].names, weights.meta["categories"])
    if val_samples:
        rows = rows.shuffle(seed=seed).select(range(min(val_samples, len(rows))))
    if not len(rows):
        raise ValueError("Empty validation split")
    validation = Images(rows, weights.transforms(), mapping)
    metadata["validation"] = {"samples": len(rows)}
    train = None
    if need_train:
        rows = load_dataset(
            "parquet",
            split="train",
            streaming=True,
            data_files={"train": files["train"]},
        )
        mapping = label_mapping(rows.features["label"].names, weights.meta["categories"])
        rows = rows.shuffle(seed=seed, buffer_size=1000)
        if train_samples:
            rows = rows.take(train_samples)
        available = sum(read_metadata(path).num_rows for path in files["train"])
        size = min(train_samples, available) if train_samples else available
        if not size:
            raise ValueError("Empty ImageNet training split")
        train = TrainingImages(rows, weights.transforms(), mapping, size)
        metadata["train"] = {"samples": size, "streaming": True}
    return train, validation, metadata


@torch.no_grad()
def evaluate(model, loader, device, *, progress_every=0):
    """Sample-weighted CE/top-1/top-5 over all 1000 logits; restore module modes."""
    modes = [(module, module.training) for module in model.modules()]
    loss, top1, top5, count = 0.0, 0, 0, 0
    try:
        model.eval()
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss += F.cross_entropy(logits, labels, reduction="sum").item()
            predictions = logits.topk(min(5, logits.shape[1]), dim=1).indices
            matches = predictions.eq(labels[:, None])
            top1 += matches[:, 0].sum().item()
            top5 += matches.any(dim=1).sum().item()
            count += labels.numel()
            if progress_every and batch_index % progress_every == 0:
                print(f"Evaluated {count} images; top1={100 * top1 / count:.3f}%", flush=True)
    finally:
        for module, training in modes:
            module.training = training
    if not count:
        raise ValueError("Cannot evaluate an empty validation loader")
    return {
        "samples": count,
        "loss": loss / count,
        "top1": 100 * top1 / count,
        "top5": 100 * top5 / count,
    }


def train_epoch(
    model, loader, optimizer, device, regularizer=None, strength=0.0, *, after_step=None
):
    """Train on the training split only; sparse loss stays a scalar autograd term."""
    model.train()
    total, sparse_total, count = 0.0, 0.0, 0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        task_loss = F.cross_entropy(model(images), labels)
        sparse_loss = regularizer() if regularizer is not None else task_loss.new_zeros(())
        (task_loss + strength * sparse_loss).backward()
        optimizer.step()
        if after_step is not None:
            after_step(batch_index)
        count += labels.numel()
        total += task_loss.detach().item() * labels.numel()
        sparse_total += sparse_loss.detach().item() * labels.numel()
    if not count:
        raise ValueError("Cannot train on an empty loader")
    return {"task_loss": total / count, "sparse_loss": sparse_total / count}
