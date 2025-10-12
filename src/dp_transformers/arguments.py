# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional

import numpy as np
from scipy import optimize
from transformers import TrainingArguments as HfTrainingArguments
from transformers import IntervalStrategy, logging
from dataclasses import dataclass, field
from datasets.utils import disable_progress_bar

from scipy import optimize as opt

from dp_accounting import dp_event as event
from dp_accounting.pld import pld_privacy_accountant as pld
from dp_accounting.rdp import rdp_privacy_accountant as rdp

logger = logging.get_logger(__name__)


@dataclass
class PrivacyArguments:
    per_sample_max_grad_norm: Optional[float] = field(default=None, metadata={"help": "Max per sample clip norm for private gradients"})
    public_clip: Optional[float] = field(default=float("inf"), metadata={"help": "Max per sample clip norm for public gradients"})
    noise_multiplier: Optional[float] = field(default=None, metadata={"help": "Noise multiplier for DP training"})
    target_epsilon: Optional[float] = field(default=None, metadata={
        "help": "Target epsilon at end of training (mutually exclusive with noise multiplier)"
    })
    target_delta: Optional[float] = field(default=None, metadata={
        "help": "Target delta, defaults to 1/N"
    })
    disable_dp: bool = field(default=False, metadata={
        "help": "Disable DP training."
    })
    selective_dp: bool = field(default=False, metadata={
        "help": "Enable Selective DP Algorithm"
    })

    def initialize(self, sampling_probability: float, num_steps: int, num_samples: int) -> None:
        if self.target_delta is None:
            self.target_delta = 1.0/num_samples
        logger.info(f"The target delta is set to be: {self.target_delta}")

        # Set up noise multiplier
        if self.noise_multiplier is None:
            self.noise_multiplier = find_noise_multiplier(
                sampling=sampling_probability,
                step=num_steps,
                target_delta=self.target_delta,
                target_epsilon=self.target_epsilon
            )
        logger.info(f"The noise multiplier is set to be: {self.noise_multiplier}")

    @property
    def is_initialized(self) -> bool:
        return (
            self.per_sample_max_grad_norm is not None and
            self.noise_multiplier is not None and
            self.target_delta is not None
        )

    def __post_init__(self):
        if self.disable_dp:
            logger.warning("Disabling differentially private training...")
            self.noise_multiplier = 0.0
            self.per_sample_max_grad_norm = float('inf')
            self.target_epsilon = None
        else:
            if bool(self.target_epsilon) == bool(self.noise_multiplier):
                raise ValueError("Exactly one of the arguments --target_epsilon and --noise_multiplier must be used.")
            if self.per_sample_max_grad_norm is None:
                raise ValueError("DP training requires --per_sample_max_grad_norm argument.")


@dataclass
class TrainingArguments(HfTrainingArguments):
    dry_run: bool = field(
        default=False,
        metadata={"help": "Option for reducing training steps (2) and logging intervals (1) for quick sanity checking of arguments."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.dry_run:
            logger.warning("--dry_run was specified. Reducing number of training steps to 2 and logging intervals to 1...")
            self.logging_steps = 1
            self.logging_strategy = IntervalStrategy.STEPS
            self.eval_steps = 1
            self.evaluation_strategy = IntervalStrategy.STEPS

            self.max_steps = 2

        if self.disable_tqdm:
            disable_progress_bar()


def find_noise_multiplier(sampling: float, step: int, target_epsilon: float, target_delta: float,) -> float:
    """
    Find a noise multiplier that satisfies a given target epsilon.

    :param float sampling_probability: Probability of a record being in batch for Poisson sampling
    :param int num_steps: Number of optimisation steps
    :param float target_epsilon: Desired target epsilon
    :param float target_delta: Value of DP delta
    :param float eps_error: Error allowed for final epsilon
    """
    RDP_ORDERS = (
        [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
        + list(range(5, 64))
        + [128, 256, 512]
    )
    def objective(noise_multiplier):
        accountant = rdp.RdpAccountant(RDP_ORDERS)
        dpevent = event.SelfComposedDpEvent(
            event.PoissonSampledDpEvent(
                sampling, event.GaussianDpEvent(noise_multiplier)
            ),
            step,
        )
        accountant.compose(dpevent)
        eps = accountant.get_epsilon(target_delta)
        return eps - target_epsilon

    optimal_noise = opt.brentq(objective, 1e-6, 1000)
    return optimal_noise