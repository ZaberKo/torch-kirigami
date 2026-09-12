"""Pretrained-model group regularization until channel selection stabilizes."""

from workflow_utils import Experiment, arguments

from torch_kirigami.sparsity import GroupSquaredL2, SelectionWindow


def main():
    options = arguments(__doc__)
    run = Experiment(options, training=True)
    window = SelectionWindow(2)
    stable = False
    epochs = max(3, options.sparse_epochs)
    for epoch in range(epochs):
        plan = run.plan()
        chosen = set(plan.selected)
        retained = {}
        for axis in run.space.axes:
            removed = {
                index
                for candidate in run.space.candidates
                if candidate.axis == axis and candidate.key in chosen
                for index in candidate.remove[0].fully_selected_indices(0)
            }
            retained[axis.tensor.paths[0]] = sorted(set(range(axis.tensor.shape[0])) - removed)
        similarity = window.update(retained)
        if similarity is not None and similarity >= 0.99:
            stable = True
            break
        groups = run.groups(plan)
        regularizer = GroupSquaredL2(groups) if groups else None
        run.train(regularizer=regularizer, strength=options.strength * (epoch + 1) / epochs)
    run.record("search_completed", stable=stable, epochs=epoch + 1)
    run.prune(run.plan())
    run.finetune()
    window.reset()
    run.finish(stable=stable, window=window.state_dict())


if __name__ == "__main__":
    main()
