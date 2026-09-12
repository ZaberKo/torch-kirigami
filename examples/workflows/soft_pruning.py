"""Soft zeroing or norm decay of pretrained-model groups before physical pruning."""

from workflow_utils import Experiment, arguments

from torch_kirigami.sparsity import GroupLasso, set_group_norms_, zero_groups_


def configure(parser):
    parser.add_argument("--operation", choices=("zero", "decay"), default="decay")


def main():
    options = arguments(__doc__, configure=configure)
    if options.sparse_epochs < 1:
        raise ValueError("Soft pruning requires at least one sparse epoch")
    run = Experiment(options, training=True)
    # Establish momentum before zeroing both sides of a complete dependency group.
    run.train()
    run.record("warmup_trained")
    for cycle in range(2):
        plan = run.plan()
        groups = run.union_group(plan)
        initial_norm = GroupLasso(groups)().detach().item() if groups else 0.0
        batches = len(run.train_loader)
        for epoch in range(options.sparse_epochs):

            def project(
                batch, epoch=epoch, groups=groups, initial_norm=initial_norm, batches=batches
            ):
                if not groups:
                    return
                if options.operation == "zero":
                    zero_groups_(groups)
                else:
                    progress = (epoch * batches + batch) / (options.sparse_epochs * batches)
                    set_group_norms_(groups, (initial_norm * (1 - progress),))

            run.train(after_step=project)
        run.record(f"cycle_{cycle + 1}_projected")
        if cycle == 0:
            # Unprojected updates let retained SGD momentum regrow zeroed regions.
            run.train()
    run.prune(plan)
    run.finetune()
    run.finish(operation=options.operation, cycles=2)


if __name__ == "__main__":
    main()
