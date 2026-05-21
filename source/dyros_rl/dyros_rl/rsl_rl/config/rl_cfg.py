from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg


########################################################
# Runner Configuration                                 #
########################################################
@configclass
class DyrosRslRlPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    algorithm: RslRlPpoAlgorithmCfg | DyrosRslRlPpoAlgorithmCfg = MISSING

########################################################
# Algorithm Configuration                             #
########################################################
@configclass
class DyrosRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    bound_loss_cfg: RslRlBoundLossCfg | None = None
    lcp_loss_cfg: RslRlLcpLossCfg | None = None

########################################################
# Loss Configuration                                   #
########################################################
@configclass
class RslRlBoundLossCfg:
    """Configuration for the bound loss."""
    bound_loss_coef: float = 10.0
    """The coefficient for the bound loss."""

    bound_range: float = 1.1
    """The range for the bound loss."""

@configclass
class RslRlLcpLossCfg:
    """Configuration for the LCP module."""

    gradient_penalty_coef: float = 0.0
    """The coefficient for the gradient penalty loss."""

    is_lcp: bool = True
    """Whether to use the LCP module."""

    gradient_penalty_coef_schedule: list[float] = MISSING
    """The schedule for the gradient penalty coefficient."""

    gradient_penalty_coef_schedule_steps: list[int] = MISSING
    """The steps for the gradient penalty coefficient schedule."""
