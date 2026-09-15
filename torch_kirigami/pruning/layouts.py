"""Shape proofs valid independently of execution backend and tensor strides."""


def view_preserves_stride_boundaries(
    source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> bool:
    """Prove a view valid without knowing any stride values.

    Singleton axes may be inserted or removed. Each remaining source axis may
    be split into consecutive factors, but adjacent non-singleton source axes
    cannot be merged: their strides need not satisfy view's contiguous-subspace
    condition. Splitting one axis needs only that axis's own stride, including
    zero strides, so this proof also covers overlapping and expanded tensors.

    Args:
        source_shape: Positive, concrete input dimensions.
        target_shape: Positive, concrete output dimensions, with `-1` resolved.

    Returns:
        Whether the transformation is valid for every legal input stride.
    """
    if any(size <= 0 for size in (*source_shape, *target_shape)):
        return False
    targets = iter(size for size in target_shape if size != 1)
    for size in source_shape:
        while size != 1:
            factor = next(targets, None)
            if factor is None or size % factor:
                return False
            size //= factor
    return next(targets, None) is None
