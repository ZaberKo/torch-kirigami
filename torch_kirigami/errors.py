"""Explicit failures at capture, analysis, and lifecycle boundaries."""


class KirigamiError(Exception):
    """Base library error."""


class CaptureError(KirigamiError):
    """The model could not be captured or safely executed for metadata."""


class AnalysisLimitError(KirigamiError):
    """An exact symbolic selection exceeded the supported complexity budget."""


class StaleGraphError(KirigamiError):
    """The source model no longer has the captured structure."""


class UnsupportedOperation(Exception):
    """A captured operation has no proven rule for these arguments."""
