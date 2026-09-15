"""Shared local ImageNet loading, label alignment and held-out evaluation."""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL.Image import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import WeightsEnum
from tqdm.auto import tqdm


def label_mapping(names: Sequence[str], categories: Sequence[str]) -> tuple[int, ...]:
    """Verify HF label indices match torchvision output indices; return identity.

    HF names may list several comma-separated synonyms. The index-specific
    aliases below account only for display-name differences in ImageNet-1k;
    this function rejects mismatches rather than reordering dataset labels.
    """
    if len(names) != 1000 or len(categories) != 1000:
        raise ValueError("Expected the original 1000 ImageNet classes")
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


class Images(Dataset[tuple[torch.Tensor, int]]):
    """Read either split from Arrow, decoding with the weights' PIL preprocessing.

    DataLoader workers perform decoding and transforms; their PyTorch intra-op
    thread count is one. Batched fetches avoid repeated HF row-formatting calls.
    """

    def __init__(
        self, rows: HFDataset, transform: Callable[[Image], torch.Tensor], mapping: Sequence[int]
    ) -> None:
        self.rows, self.transform, self.mapping = rows, transform, mapping

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[int(index)]
        return self.transform(row["image"].convert("RGB")), self.mapping[row["label"]]

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, int]]:
        """Fetch one batch through HF while preserving sample order and labels."""
        # DataLoader commonly requests a contiguous validation range. A slice
        # lets Arrow avoid gathering individual rows even for a full batch.
        if not indices:
            return []
        start = indices[0]
        stop = start + len(indices)
        key = (
            slice(start, stop)
            if 0 <= start < stop <= len(self) and indices == list(range(start, stop))
            else indices
        )
        rows = self.rows[key]
        return [
            (self.transform(image.convert("RGB")), self.mapping[label])
            for image, label in zip(rows["image"], rows["label"], strict=True)
        ]


def load_images(
    weights: WeightsEnum,
    data_dir: str | Path | None = None,
    *,
    need_train: bool = False,
    train_samples: int = 0,
    val_samples: int = 0,
    seed: int = 7,
) -> tuple[Images | None, Images, dict[str, Any]]:
    """Prepare/reuse memory-mapped HF Arrow caches from local ImageNet shards.

    Both splits use map-style datasets. A limited subset is sampled once and
    materialized in Arrow to remove indirect indices. The training DataLoader
    owns per-epoch shuffling. Cached images still contain encoded image bytes;
    decoding and deterministic preprocessing remain in DataLoader workers.
    No dataset files are downloaded. `HF_DATASETS_CACHE` selects the cache root.
    """
    if train_samples < 0 or val_samples < 0:
        raise ValueError("Sample limits must be nonnegative")
    dataset = "ILSVRC/imagenet-1k"
    splits = ("validation", "train") if need_train else ("validation",)
    root = Path(
        data_dir
        if data_dir is not None
        else snapshot_download(
            dataset,
            repo_type="dataset",
            local_files_only=True,
            allow_patterns=[f"data/{split}-*.parquet" for split in splits],
        )
    ).resolve()
    files = {
        split: [str(path) for path in sorted((root / "data").glob(f"{split}-*.parquet"))]
        for split in splits
    }
    for split in splits:
        if not files[split]:
            raise FileNotFoundError(f"Missing ImageNet {split} shards under {root / 'data'}")
    metadata = {"dataset": dataset, "data_dir": str(root), "files": files, "subset_seed": seed}
    images = {}
    for split in splits:
        print(f"[{split}] Preparing/reusing local PyArrow cache", flush=True)
        rows = load_dataset(
            "parquet", split=split, data_files={split: files[split]}, streaming=False
        )
        mapping = label_mapping(rows.features["label"].names, weights.meta["categories"])
        limit = train_samples if split == "train" else val_samples
        if limit and limit < len(rows):
            rows = rows.shuffle(seed=seed).select(range(limit)).flatten_indices()
        if not len(rows):
            raise ValueError(f"Empty ImageNet {split} split")
        images[split] = Images(rows, weights.transforms(), mapping)
        metadata[split] = {
            "samples": len(rows),
            "storage": "arrow",
            "cache_files": [item["filename"] for item in rows.cache_files],
        }
        print(f"[{split}] Arrow cache ready: {len(rows)} images", flush=True)
    return images.get("train"), images["validation"], metadata


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str,
    *,
    description: str = "Evaluation",
) -> dict[str, int | float]:
    """Measure sample-weighted CE/top-1/top-5 with batch progress and restore modes."""
    modes = [(module, module.training) for module in model.modules()]
    loss, top1, top5, count = 0.0, 0, 0, 0
    try:
        model.eval()
        with tqdm(
            loader, desc=f"{description} ({device})", unit="batch", dynamic_ncols=True
        ) as progress:
            for images, labels in progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss += F.cross_entropy(logits, labels, reduction="sum").item()
                matches = logits.topk(min(5, logits.shape[1]), dim=1).indices.eq(labels[:, None])
                top1 += matches[:, 0].sum().item()
                top5 += matches.any(dim=1).sum().item()
                count += labels.numel()
                progress.set_postfix(
                    images=count,
                    loss=f"{loss / count:.4f}",
                    top1=f"{100 * top1 / count:.3f}%",
                    refresh=False,
                )
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
