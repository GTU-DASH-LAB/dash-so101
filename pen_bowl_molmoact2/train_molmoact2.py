#!/usr/bin/env python3
"""Launcher for lerobot's own training loop, unchanged, with the Accelerator built here so the
mixed precision comes from MIXED_PRECISION (config.env): `accelerate launch` otherwise forces
"no" unless given its own flag, and lerobot passes none. MolmoAct2's docs train with bf16.

Started by watchdog_train.sh through `accelerate launch`. Not meant to be run by hand.
"""

import os

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from lerobot.scripts import lerobot_train
from lerobot.utils.import_utils import register_third_party_plugins


def main() -> None:
    register_third_party_plugins()
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # same as lerobot's own Accelerator
        mixed_precision=os.environ.get("MIXED_PRECISION", "bf16"),
        # LoRA leaves the frozen VLM weights without gradients.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, gradient_as_bucket_view=True)],
    )
    lerobot_train.train(accelerator=accelerator)


if __name__ == "__main__":
    main()
