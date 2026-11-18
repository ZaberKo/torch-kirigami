"""Exceptions raised at capture, analysis, and graph-lifecycle boundaries."""


class KirigamiError(Exception):
    """Base class for the library's capture, analysis, and lifecycle errors."""


class CaptureError(KirigamiError):
    """The model could not be captured, or could not be executed safely to collect metadata."""


class AnalysisLimitError(KirigamiError):
    """An exact symbolic selection would exceed a configured complexity limit.

    The limits are fields of `torch_kirigami.config`, such as `max_parts`.
    """


class StaleGraphError(KirigamiError):
    """A dependency graph no longer matches its source model or analysis settings.

    Rebuild the graph after structural model changes or after changing a
    structural `max_*` setting in `torch_kirigami.config`.
    """


class UnsupportedOperation(Exception):
    """An operator rule cannot prove the semantics of a captured call.

    Operator rules raise this error to decline a call or a recipe request.
    Graph construction catches it, records an `unsupported` diagnostic, and adds
    a `Barrier` over the affected tensors. Plan compilation converts it into a
    `torch_kirigami.pruning.PlanningError`. It derives from `Exception`, not
    from `KirigamiError`.
    """
