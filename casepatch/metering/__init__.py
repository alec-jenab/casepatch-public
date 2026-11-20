"""Local usage metering and opt-in, anonymous activation events.

Nothing in this package opens a network connection. Uploading events is refused unless an endpoint is both
configured and listed in APPROVED_ENDPOINTS, which is empty until an endpoint is added in a reviewed commit.
"""
from .events import (APPROVED_ENDPOINTS, EventLog, EventRejected, Settings, TelemetryDisabled, Uploader, Meter,
                     casepatch_home)
from .active import activation_metrics

__all__ = ["APPROVED_ENDPOINTS", "EventLog", "EventRejected", "Settings", "TelemetryDisabled", "Uploader", "Meter",
           "casepatch_home", "activation_metrics"]
