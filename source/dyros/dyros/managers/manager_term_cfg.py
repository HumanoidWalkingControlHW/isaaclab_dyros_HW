from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

@configclass
class DyrosObservationGroupCfg(ObservationGroupCfg):
    """Configuration for a Dyros observation group."""

    skip_history_tick: int = 2
    """The number of history ticks to skip. Defaults to 2."""

@configclass
class DyrosObservationTermCfg(ObservationTermCfg):
    """Configuration for a Dyros observation term."""

    skip_history_tick: int = 2
    """The number of history ticks to skip. Defaults to 2."""