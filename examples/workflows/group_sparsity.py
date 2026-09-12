"""Pretrained CNN/ViT dependency-group Lasso or growing squared-L2 training."""

from workflow_utils import Experiment, arguments

from torch_kirigami.sparsity import GroupLasso, GroupSquaredL2


def configure(parser):
    parser.add_argument("--penalty", choices=("lasso", "squared"), default="lasso")


def main():
    options = arguments(__doc__, configure=configure)
    run = Experiment(options, training=options.sparse_epochs > 0)
    regularizer = (
        GroupLasso(run.groups()) if options.penalty == "lasso" and options.sparse_epochs else None
    )
    for epoch in range(options.sparse_epochs):
        strength = options.strength
        if options.penalty == "squared":
            groups = run.groups(run.plan())
            regularizer = GroupSquaredL2(groups) if groups else None
            strength *= (epoch + 1) / options.sparse_epochs
        run.train(regularizer=regularizer, strength=strength)
    if options.sparse_epochs:
        run.record("sparse_trained")
    run.prune(run.plan())
    run.finetune()
    run.finish(penalty=options.penalty)


if __name__ == "__main__":
    main()
