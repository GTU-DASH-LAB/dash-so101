#!/usr/bin/env python3
"""Thin launcher for lerobot's own training loop (lerobot.scripts.lerobot_train).

It runs exactly `lerobot-train`, same CLI and same loop, with one change: we build
the `Accelerator` ourselves and hand it to `train(accelerator=...)`, a public argument
of lerobot's train(). That lets us tune multi-GPU communication for THIS machine:

* The two RTX 5090s have no NVLink and no P2P, so every gradient all-reduce goes
  GPU -> system RAM -> GPU, and GPU1 sits on a PCIe x4 link. Measured with lerobot's
  default Accelerator (fp32 gradients, find_unused_parameters=True), DDP added
  ~56 ms to every step.
* comm_hook=BF16 sends gradients as bf16 (half the bytes). The optimizer still
  updates fp32 master weights. This is a standard PyTorch DDP hook.
* find_unused_parameters=False skips a full autograd-graph scan every step. ACT
  uses all of its parameters in training (VAE encoder included), so the scan
  only costs time. If that ever stops being true, DDP raises a clear error at
  step 2 rather than training wrongly.
* mixed_precision comes from MIXED_PRECISION (config.env) instead of the
  ACCELERATE_MIXED_PRECISION env var. lerobot passes mixed_precision=None for ACT,
  and `accelerate launch` then forces "no" unless given its own flag.

Started by watchdog_train.sh through `accelerate launch`. Not meant to be run by hand.
"""

import os

from accelerate import Accelerator
from accelerate.utils import DDPCommunicationHookType, DistributedDataParallelKwargs

from lerobot.scripts import lerobot_train
from lerobot.utils.import_utils import register_third_party_plugins


def main() -> None:
    register_third_party_plugins()
    multi_gpu = int(os.environ.get("WORLD_SIZE", "1")) > 1
    ddp = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        comm_hook=DDPCommunicationHookType.BF16 if multi_gpu else DDPCommunicationHookType.NO,
    )
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # same as lerobot's own Accelerator
        mixed_precision=os.environ.get("MIXED_PRECISION", "bf16"),
        kwargs_handlers=[ddp],
    )
    lerobot_train.train(accelerator=accelerator)


if __name__ == "__main__":
    main()
