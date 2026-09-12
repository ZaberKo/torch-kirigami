"""Pretrained ResNet BN-scale L1 training, scale-ranked pruning and fine-tuning."""

from workflow_utils import Experiment, arguments

from torch_kirigami.sparsity import ScaleL1


def main():
    options = arguments(__doc__)
    if options.model != "resnet18":
        raise ValueError("BN sparsity requires resnet18; ViT uses LayerNorm")
    run = Experiment(options, training=options.sparse_epochs > 0)
    if options.sparse_epochs:
        regularizer = ScaleL1(run.space.graph, run.scale_paths("bn"))
        run.train(options.sparse_epochs, regularizer=regularizer, strength=options.strength)
        run.record("sparse_trained")
    run.prune(run.plan(metric="bn"))
    run.finetune()
    run.finish()


if __name__ == "__main__":
    main()
