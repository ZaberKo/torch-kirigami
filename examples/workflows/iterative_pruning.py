"""Repeated pretrained-model pruning with cumulative channel budgets and fine-tuning."""

from workflow_utils import Experiment, arguments


def main():
    options = arguments(__doc__, rounds=3)
    run = Experiment(options)
    for index in range(1, options.rounds + 1):
        ratio = options.ratio * index / options.rounds
        run.prune(run.plan(ratio), stage=f"round_{index}_pruned", target_ratio=ratio)
        run.finetune(stage=f"round_{index}_finetuned")
    run.finish(completed_rounds=options.rounds)


if __name__ == "__main__":
    main()
