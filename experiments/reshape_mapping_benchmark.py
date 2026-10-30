"""Measure exact split-batch reshape mapping without allocating activation tensors.

Run `PYTHONPATH=. .venv/bin/python experiments/reshape_mapping_benchmark.py
--output result.json` against each checkout. The seed selects two feature
positions. Batch and patch coordinates must remain symbolic in the result.
"""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from torch_kirigami import AnalysisLimitError, ReshapeRelation, TensorRef, config


def measure(batch: int) -> dict[str, object]:
    """Time three mappings and check resolved coordinates against axis selections."""
    source = TensorRef("source", (batch * 7, 42, 128))
    target = TensorRef("target", (batch, 7, 42, 128))
    relation = ReshapeRelation(source, target)
    seed = source.axis(-1).select([0, 7])
    timings = []
    result: dict[str, object] = {}
    for _ in range(3):
        started = perf_counter()
        try:
            mapped = relation.propagate(seed)[0]
        except AnalysisLimitError as error:
            result = {"status": "analysis_limit", "message": str(error)}
        else:
            result = {
                "status": "resolved",
                "regions": len(mapped.regions),
                "selected_elements": mapped.count,
            }
        timings.append(perf_counter() - started)
        if result["status"] == "resolved":
            assert mapped == target.axis(-1).select([0, 7])
            assert relation.propagate(mapped)[0] == seed
    return dict(result, median_seconds=median(timings))


def main() -> None:
    """Record batch-one and full-batch cases using the same expansion limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "max_parts": config.max_parts,
        "cases": {str(batch): measure(batch) for batch in (1, 128)},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
