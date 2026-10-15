#!/usr/bin/env python3
"""Launcher for lerobot's own training loop (lerobot.scripts.lerobot_train), unchanged
except for three things, all applied before lerobot builds the dataset and the model:

1. Video window with a stride (VIDEO_FRAME_STRIDE). VLA-JEPA loads frames t..t+7 for its
   world model. That suits 5 Hz data; at our 30 Hz those 8 frames cover 0.27 s and barely
   differ. With stride 4 they cover ~1 s, so predicting the next latent frame is a real task.

2. Current state, not future state. lerobot v0.6.0's VLA-JEPA feeds the action head
   `state[:, -1]`: the LAST step of the observation window, i.e. the joint state ~1 s in the
   future (0.23 s even without our stride). At inference only the current state exists, so
   the policy would train on information it never gets. We pass `state[:, 0]` (time t).

3. Same image size in training as at inference. At inference the policy resizes images to
   `resize_images_to` (224x224) before Qwen; in training v0.6.0 passes them at full 480x640.
   We resize in training too (area interpolation, on the GPU), which also matches
   pretraining and cuts Qwen's vision tokens ~6x.

4. Gradient checkpointing for Qwen3-VL (GRADIENT_CHECKPOINTING=true). A full fine-tune of
   2.77B parameters keeps ~22 GB of weights, gradients and AdamW state on each 32 GB card;
   without checkpointing, batch 8 ran out of memory at step 2 (measured). Activations are
   recomputed in the backward pass instead of stored (non-reentrant variant, which is the
   one that is correct with frozen sub-modules).

Inference needs none of this: a published checkpoint runs on stock lerobot (`select_action`
only sees the current frame and the current state, and resizes images itself).

Also builds the Accelerator: mixed_precision from MIXED_PRECISION ("no": the model is held
in bf16 already), find_unused_parameters=True because parts of the model take no
gradient (frozen V-JEPA encoder, Qwen's final norm), and gradient_as_bucket_view=True.

Started by watchdog_train.sh through `accelerate launch`. Not meant to be run by hand.
"""

import os

import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from lerobot.policies.vla_jepa import configuration_vla_jepa, modeling_vla_jepa
from lerobot.scripts import lerobot_train
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins

STRIDE = int(os.environ.get("VIDEO_FRAME_STRIDE", "1"))
GRAD_CKPT = os.environ.get("GRADIENT_CHECKPOINTING", "false") == "true"


def _observation_delta_indices(self) -> list[int]:
    return list(range(0, self.num_video_frames * STRIDE, STRIDE))


_orig_prepare = modeling_vla_jepa.VLAJEPAPolicy._prepare_model_inputs


def _prepare_model_inputs(self, batch, training=True):
    batch = dict(batch)
    size = self.config.resize_images_to
    if size is not None:
        size = tuple(size)
        for key in self.config.image_features:
            t = batch[key]
            if tuple(t.shape[-2:]) != size:
                lead = t.shape[:-3]
                flat = F.interpolate(t.reshape(-1, *t.shape[-3:]).float(), size=size, mode="area")
                batch[key] = flat.reshape(*lead, t.shape[-3], *size)
    state = batch.get(OBS_STATE)
    if state is not None and state.ndim > 2:
        batch[OBS_STATE] = state[:, 0]
    return _orig_prepare(self, batch, training)


_orig_model_init = modeling_vla_jepa.VLAJEPAModel.__init__


def _model_init(self, config) -> None:
    _orig_model_init(self, config)
    if GRAD_CKPT:
        self.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.qwen.model.config.use_cache = False


def main() -> None:
    configuration_vla_jepa.VLAJEPAConfig.observation_delta_indices = property(_observation_delta_indices)
    modeling_vla_jepa.VLAJEPAPolicy._prepare_model_inputs = _prepare_model_inputs
    modeling_vla_jepa.VLAJEPAModel.__init__ = _model_init
    register_third_party_plugins()
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # same as lerobot's own Accelerator
        mixed_precision=os.environ.get("MIXED_PRECISION", "no"),
        # gradient_as_bucket_view: DDP's all-reduce buckets ARE the gradients instead of a
        # second 5.5 GB copy of them, which would not fit next to the 29 GB used on 1 GPU.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, gradient_as_bucket_view=True)],
    )
    lerobot_train.train(accelerator=accelerator)


if __name__ == "__main__":
    main()
