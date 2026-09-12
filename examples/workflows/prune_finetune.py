"""Pretrained model -> magnitude/Taylor pruning -> optional ImageNet fine-tuning."""

from workflow_utils import Experiment, arguments


def configure(parser):
    parser.add_argument("--metric", choices=("magnitude", "taylor"), default="magnitude")


def main():
    options = arguments(__doc__, configure=configure)
    run = Experiment(options, calibration=options.metric == "taylor")
    if options.metric == "taylor":
        run.task_gradients()
    run.prune(run.plan(metric=options.metric))
    run.finetune()
    run.finish()


if __name__ == "__main__":
    main()
