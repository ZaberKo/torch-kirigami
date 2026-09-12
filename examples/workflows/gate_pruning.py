"""Identity gates on pretrained CNN channels or ViT FFN units, then L1 and pruning."""

from workflow_utils import Experiment, arguments

from torch_kirigami.sparsity import ScaleL1


def main():
    options = arguments(__doc__)
    run = Experiment(options, training=options.sparse_epochs > 0, gated=True)
    if options.sparse_epochs:
        regularizer = ScaleL1(run.space.graph, run.scale_paths("gate"))
        run.train(options.sparse_epochs, regularizer=regularizer, strength=options.strength)
        run.record("gate_trained")
    run.prune(run.plan(metric="gate"))
    run.finetune()
    run.finish()


if __name__ == "__main__":
    main()
