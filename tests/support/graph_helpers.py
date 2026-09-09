"""support / graph helpers contracts."""


def indices(impact, tensor, dim):
    return set(impact.selection(tensor).fully_selected_indices(dim))


def removed(impact, tensor, dim):
    return set(impact.selection(tensor).fully_selected_indices(dim))
