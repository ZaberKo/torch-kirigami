"""ImageNet labels, Arrow cache reuse, worker fetching, and training views."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")


import imagenet_data as imagenet
from datasets import ClassLabel, Dataset, Features, Image
from huggingface_hub import HfApi, constants
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image as PILImage
from torchvision.models import ResNet18_Weights


def test_imagenet_label_order_and_synonyms():
    categories = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    names = [f"{category}, synonym {i}" for i, category in enumerate(categories)]
    assert imagenet.label_mapping(names, categories) == tuple(range(1000))
    names[134], names[517], names[639] = "crane", "crane2", "maillot, tank suit"
    assert imagenet.label_mapping(names, categories) == tuple(range(1000))
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ValueError, match="class order"):
        imagenet.label_mapping(names, categories)
    with pytest.raises(ValueError, match="1000"):
        imagenet.label_mapping(names[:10], categories)


@pytest.mark.parametrize("need_train", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
def test_data_loading_requests_only_needed_splits(tmp_path, monkeypatch, need_train, use_cache):
    weights = ResNet18_Weights.IMAGENET1K_V1
    features = Features({"image": Image(), "label": ClassLabel(names=weights.meta["categories"])})
    folder = tmp_path / "data"
    folder.mkdir()
    for split in ("validation", "train") if need_train else ("validation",):
        labels = [0, 1, 2, 3] if split == "validation" else [4, 5, 6, 7]
        rows = Dataset.from_dict(
            {"image": [PILImage.new("L", (8, 8))] * 4, "label": labels}, features=features
        )
        rows.to_parquet(str(folder / f"{split}-00000.parquet"))
    data_dir = tmp_path
    if use_cache:
        cache = tmp_path / "hub"
        repo = cache / "datasets--ILSVRC--imagenet-1k"
        revision = "a" * 40
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "data").symlink_to(folder, target_is_directory=True)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(revision)
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
        data_dir = None

    def no_network(*args, **kwargs):
        pytest.fail("Pre-downloaded ImageNet must be resolved without Hub requests")

    monkeypatch.setattr(HfApi, "repo_info", no_network)
    train, validation, metadata = imagenet.load_images(
        weights, data_dir, need_train=need_train, train_samples=2, val_samples=3
    )
    assert len(validation) == 3
    assert validation[0][0].shape == (3, 224, 224)
    assert metadata["validation"]["samples"] == 3
    assert metadata["validation"]["storage"] == "arrow"
    assert metadata["validation"]["cache_files"]
    assert validation.rows._indices is None  # Subsets are physically materialized.
    if need_train:
        samples = list(train)
        assert len(samples) == len(train) == 2
        assert isinstance(train, imagenet.Images)
        assert metadata["train"]["storage"] == "arrow"
        assert metadata["train"]["cache_files"]
        assert {label for _, label in samples}.isdisjoint({validation[i][1] for i in range(3)})
        assert set(metadata["files"]) == {"validation", "train"}
        full_train, full_validation, full_metadata = imagenet.load_images(
            weights, data_dir, need_train=True
        )
        assert len(list(full_train)) == len(full_train) == 4
        assert len(full_validation) == full_metadata["train"]["samples"] == 4
    else:
        assert train is None and set(metadata["files"]) == {"validation"}
        with pytest.raises(FileNotFoundError, match="train"):
            imagenet.load_images(weights, data_dir, need_train=True)


def test_missing_hf_cache_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty-hub"))

    def no_network(*args, **kwargs):
        pytest.fail("Missing cached data must not trigger a download")

    monkeypatch.setattr(HfApi, "repo_info", no_network)
    with pytest.raises(LocalEntryNotFoundError):
        imagenet.load_images(ResNet18_Weights.IMAGENET1K_V1)


@pytest.mark.parametrize("indices", [[0, 1, 2], [2, 0, 2], [-1, 0], []])
def test_arrow_image_batch_fetch_matches_scalar_preprocessing(indices):
    weights = ResNet18_Weights.IMAGENET1K_V1
    pictures = [
        PILImage.new(mode, size, color)
        for mode, size, color in (
            ("RGB", (300, 260), (10, 60, 200)),
            ("L", (150, 350), 120),
            ("RGB", (500, 180), (255, 0, 70)),
        )
    ]
    rows = Dataset.from_dict(
        {"image": pictures, "label": [0, 1, 2]},
        features=Features({"image": Image(), "label": ClassLabel(names=["a", "b", "c"])}),
    )
    images = imagenet.Images(rows, weights.transforms(), (2, 0, 1))
    batch = images.__getitems__(indices)
    assert len(batch) == len(indices)
    for (tensor, label), index in zip(batch, indices, strict=True):
        torch.testing.assert_close(
            tensor, weights.transforms()(pictures[index].convert("RGB")), rtol=0, atol=0
        )
        assert label == (2, 0, 1)[index]
    with pytest.raises(IndexError):
        images.__getitems__([2, 3])


def test_arrow_caches_reuse_and_spawn_workers(tmp_path):
    weights = ResNet18_Weights.IMAGENET1K_V1
    features = Features({"image": Image(), "label": ClassLabel(names=weights.meta["categories"])})
    folder = tmp_path / "data"
    folder.mkdir()
    pictures = [PILImage.new("RGB", (250, 280), (i * 30, 15, 200)) for i in range(5)]
    for split in ("validation", "train"):
        rows = Dataset.from_dict({"image": pictures, "label": list(range(5))}, features=features)
        rows.to_parquet(folder / f"{split}-00000.parquet")
    train, validation, metadata = imagenet.load_images(
        weights, tmp_path, need_train=True, val_samples=3
    )
    paths = [
        Path(path) for split in ("train", "validation") for path in metadata[split]["cache_files"]
    ]
    before = [(path, path.stat().st_mtime_ns) for path in paths]
    _, repeated, repeated_metadata = imagenet.load_images(
        weights, tmp_path, need_train=True, val_samples=3
    )
    assert repeated_metadata == metadata
    assert [(path, path.stat().st_mtime_ns) for path in paths] == before
    assert [repeated[i][1] for i in range(3)] == [validation[i][1] for i in range(3)]
    parent_threads = torch.get_num_threads()
    # Map-style Arrow data must survive spawn and persistent worker reuse. Each
    # shuffled traversal visits all samples exactly once, unlike an unsharded iterable.
    for dataset, shuffle in ((train, True), (validation, False)):
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            persistent_workers=True,
            multiprocessing_context="spawn",
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(7),
        )
        expected_labels = sorted(dataset[i][1] for i in range(len(dataset)))
        for _ in range(2):
            seen = []
            for tensors, labels in loader:
                for tensor, label in zip(tensors, labels.tolist(), strict=True):
                    torch.testing.assert_close(
                        tensor, weights.transforms()(pictures[label]), rtol=0, atol=0
                    )
                    seen.append(label)
            assert sorted(seen) == expected_labels
        del loader
    assert torch.get_num_threads() == parent_threads


def test_training_view_shares_arrow_rows_and_preserves_calibration_transform():
    weights = ResNet18_Weights.IMAGENET1K_V1
    picture = PILImage.new("RGB", (300, 260), (10, 60, 200))
    rows = Dataset.from_dict({"image": [picture], "label": [0]})
    calibration = imagenet.Images(rows, weights.transforms(), (0,))
    before, label = calibration[0]
    training = calibration.training_view(weights)
    assert training.rows is calibration.rows
    assert training.mapping is calibration.mapping
    assert training.transform is not calibration.transform
    actual, actual_label = training[0]
    assert actual.shape == before.shape == (3, 224, 224)
    assert actual_label == label == 0
    torch.testing.assert_close(calibration[0][0], before, rtol=0, atol=0)
