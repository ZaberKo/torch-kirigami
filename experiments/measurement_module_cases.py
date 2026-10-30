"""Exercise concrete nn.Module classes missing from PyTorch ModuleInfo.

Use --device cpu or cuda and --output result.jsonl. Lazy modules are explicitly
initialized before measurement. Abstract classes and registration containers
have no standalone forward and are listed separately in the audit report.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from torch_kirigami.measurement import calculate_model_complexity


def module_cases(device: str) -> list[tuple[str, nn.Module, tuple[torch.Tensor, ...]]]:
    """Build deterministic small inputs for concrete public module classes."""
    with torch.device(device):
        x = torch.randn(2, 4)
        image = torch.randn(2, 4, 6, 6)
        cases = [
            (nn.AdaptiveLogSoftmaxWithLoss(4, 8, [3, 5]), (x, torch.tensor([0, 6]))),
            (nn.ChannelShuffle(2), (image,)),
            (nn.CosineSimilarity(), (x, x)),
            (nn.CrossMapLRN2d(3), (image,)),
            (nn.EmbeddingBag(8, 4), (torch.tensor([1, 2, 3]), torch.tensor([0, 2]))),
            (nn.Flatten(), (image,)),
            (nn.Fold((4, 4), 2), (torch.randn(2, 8, 9),)),
            (nn.Hardsigmoid(), (x,)),
            (nn.Identity(), (x,)),
            (
                nn.NLLLoss2d(),
                (torch.randn(2, 3, 2, 2).log_softmax(1), torch.zeros(2, 2, 2, dtype=torch.int64)),
            ),
            (nn.PairwiseDistance(), (x, x)),
            (nn.PixelShuffle(2), (image,)),
            (nn.PixelUnshuffle(2), (image,)),
            (nn.RReLU(), (x,)),
            (nn.Sequential(nn.Linear(4, 5), nn.ReLU()), (x,)),
            (nn.SyncBatchNorm(4), (x,)),
            (nn.TripletMarginLoss(), (x, x, x)),
            (nn.TripletMarginWithDistanceLoss(), (x, x, x)),
            (nn.Unflatten(1, (2, 2)), (x,)),
            (nn.Unfold(2), (image,)),
            (nn.Upsample(scale_factor=2), (image,)),
            (nn.UpsamplingBilinear2d(scale_factor=2), (image,)),
            (nn.UpsamplingNearest2d(scale_factor=2), (image,)),
            (
                nn.TransformerDecoder(nn.TransformerDecoderLayer(4, 2, 8, batch_first=True), 2),
                (torch.randn(2, 3, 4), torch.randn(2, 5, 4)),
            ),
        ]
        if device == "cuda":
            cases.append(
                (nn.DataParallel(nn.Linear(4, 3), device_ids=[torch.cuda.current_device()]), (x,))
            )
        cases = [(type(m).__name__, m, args) for m, args in cases]
        for cls in (
            nn.AlphaDropout,
            nn.FeatureAlphaDropout,
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
        ):
            inp = (
                torch.randn(2, 4, 3, 3, 3)
                if cls == nn.Dropout3d
                else image
                if cls == nn.Dropout2d
                else torch.randn(2, 4, 3)
                if cls == nn.Dropout1d
                else x
            )
            cases.append((cls.__name__, cls(), (inp,)))
        for cls in (
            nn.LazyBatchNorm1d,
            nn.LazyBatchNorm2d,
            nn.LazyBatchNorm3d,
            nn.LazyInstanceNorm1d,
            nn.LazyInstanceNorm2d,
            nn.LazyInstanceNorm3d,
            nn.LazyLinear,
        ):
            inp = (
                torch.randn(2, 4, 3, 3, 3)
                if cls in (nn.LazyBatchNorm3d, nn.LazyInstanceNorm3d)
                else image
                if cls in (nn.LazyBatchNorm2d, nn.LazyInstanceNorm2d)
                else torch.randn(2, 4, 3)
                if cls == nn.LazyInstanceNorm1d
                else x
            )
            m = cls(3) if cls == nn.LazyLinear else cls()
            m.eval()(inp)
            cases.append((cls.__name__, m, (inp,)))
        for dims in (1, 2, 3):
            data = torch.randn(2, 4, *([6] * dims))
            pooled, indices = getattr(torch.nn.functional, f"max_pool{dims}d")(
                data, 2, return_indices=True
            )
            cases.append(
                (f"MaxUnpool{dims}d", getattr(nn, f"MaxUnpool{dims}d")(2), (pooled, indices))
            )
    return cases


def main() -> None:
    """Keep baseline forward failures distinct from measurement failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    with args.output.open("w") as output:
        for name, model, samples in module_cases(args.device):
            row = {"name": name, "device": args.device}
            stage = "baseline"
            try:
                model.eval()
                with torch.no_grad():
                    model(*samples)
                stage = "measurement"
                result = calculate_model_complexity(model, samples, device=args.device)
                row.update(status="measured", **result.__dict__)
            except Exception as error:  # noqa: BLE001 - Audit all failures independently.
                row.update(status=stage + "_error", error=str(error).split("\n", 1)[0])
            output.write(json.dumps(row) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
