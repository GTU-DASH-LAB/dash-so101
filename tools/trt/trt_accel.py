"""TensorRT acceleration of trained LeRobot policies, applied in place to a loaded policy.

    import trt_accel
    info = trt_accel.accelerate(policy)          # after .to("cuda").eval()

What is replaced, per policy type (everything else keeps running in PyTorch):

  act        the whole ACT network (ResNet-18 + transformer encoder/decoder): one engine.
  molmoact2  the flow-matching action expert: one engine for ONE denoising step (time
             conditioning, 36 blocks, the Euler update and the action-dim mask), called
             num_steps (10) times per chunk. In PyTorch this is 72 of the 126 ms of a chunk, even
             with MolmoAct2's own CUDA graph; the engine does the 10 steps in ~30 ms.

How: torch.onnx.export (dynamo) -> TensorRT's ONNX parser -> serialized plan. NOT torch_tensorrt:
its scaled_dot_product_attention converter returns wrong values whenever attn_mask is given
(even all zeros; every dtype), and its graphs of this expert trip TensorRT 10.15 Myelin checks
on sm_120. See README.md in this folder for the measurements.

Engines are cached in ~/.cache/dash_so101_trt (DASH_TRT_CACHE overrides). The cache key is a hash
of the module's weights (not file names or dates: MolmoAct2's run script recreates its deploy
folder on every run), the input shapes, the GPU and the TensorRT version. A first build takes
20-60 s; a cached engine loads in ~1 s.
"""

import hashlib
import logging
import os
import shutil
import time
from collections import deque
from pathlib import Path

import torch
from torch import nn

log = logging.getLogger("trt_accel")
CACHE = Path(os.environ.get("DASH_TRT_CACHE", Path.home() / ".cache" / "dash_so101_trt"))

_TRT2TORCH = None


def _dtype_map():
    global _TRT2TORCH
    if _TRT2TORCH is None:
        import tensorrt as trt

        _TRT2TORCH = {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16,
                      trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}
    return _TRT2TORCH


def weights_fingerprint(module, extra=""):
    """Hash of every parameter/buffer (shape + sum + abs-sum, computed on the GPU), the GPU and
    the TensorRT version. Same weights -> same key, whatever folder they were loaded from."""
    import tensorrt as trt

    h = hashlib.sha256()
    with torch.no_grad():
        for name, t in list(module.named_parameters()) + list(module.named_buffers()):
            tf = t.detach().float()
            h.update(f"{name}:{tuple(t.shape)}:{t.dtype}".encode())
            h.update(torch.stack([tf.sum(), tf.abs().sum()]).cpu().numpy().tobytes())
    h.update(torch.cuda.get_device_name().encode())
    h.update(trt.__version__.encode())
    h.update(str(extra).encode())
    return h.hexdigest()[:16]


class TRTRunner:
    """A deserialized engine called like a function on torch CUDA tensors (on the current stream)."""

    def __init__(self, plan_path):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(Path(plan_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {plan_path}")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        io = trt.TensorIOMode
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == io.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == io.OUTPUT]
        self.dynamic = {n for n in self.inputs if -1 in tuple(self.engine.get_tensor_shape(n))}
        self._out = {}
        # Own stream: on the legacy default stream TensorRT adds cudaStreamSynchronize calls.
        self.stream = torch.cuda.Stream()

    def __call__(self, *tensors):
        assert len(tensors) == len(self.inputs), (len(tensors), self.inputs)
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)                 # inputs are ready
        for name, t in zip(self.inputs, tensors):
            if name in self.dynamic:
                self.ctx.set_input_shape(name, tuple(t.shape))
            self.ctx.set_tensor_address(name, t.data_ptr())
        outs = []
        for name in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(name))
            buf = self._out.get(name)
            if buf is None or tuple(buf.shape) != shape:
                buf = torch.empty(shape, device="cuda", dtype=_dtype_map()[self.engine.get_tensor_dtype(name)])
                self._out[name] = buf
            self.ctx.set_tensor_address(name, buf.data_ptr())
            outs.append(buf)
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        for t in tensors:                              # inputs must outlive the engine's reads
            t.record_stream(self.stream)
        caller.wait_stream(self.stream)
        # The output buffers are reused by the next call: hand out copies (made on the caller's stream).
        return outs[0].clone() if len(outs) == 1 else [o.clone() for o in outs]


def build_engine(module, args, input_names, output_names, key, *, dynamic_shapes=None, profile=None,
                 strongly_typed=True, fp16=False, opt_level=3):
    """ONNX-export `module` on `args` and build a TensorRT plan, or reuse the cached one.

    profile: {input_name: (min_shape, opt_shape, max_shape)} for the inputs with dynamic dims.
    Returns (TRTRunner, "cached" | "built in N s").
    """
    import tensorrt as trt

    plan = CACHE / f"{key}.plan"
    if plan.exists():
        return TRTRunner(plan), "cached"
    t0 = time.time()
    work = CACHE / f"{key}.onnx.d"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    onnx_path = work / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(module, tuple(args), str(onnx_path), input_names=input_names,
                          output_names=output_names, dynamo=True, external_data=True, optimize=True,
                          dynamic_shapes=dynamic_shapes)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {errs}")
    cfg = builder.create_builder_config()
    cfg.builder_optimization_level = opt_level
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if fp16 and not strongly_typed:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if profile:
        prof = builder.create_optimization_profile()
        for name, (lo, opt, hi) in profile.items():
            prof.set_shape(name, lo, opt, hi)
        cfg.add_optimization_profile(prof)
    data = builder.build_serialized_network(net, cfg)
    if data is None:
        raise RuntimeError("TensorRT build failed (see the TensorRT messages above)")
    tmp = plan.with_suffix(".plan.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, plan)
    shutil.rmtree(work, ignore_errors=True)
    return TRTRunner(plan), f"built in {time.time() - t0:.0f} s"


# --------------------------------------------------------------------------------------------
# ACT
# --------------------------------------------------------------------------------------------
class _ACTCore(nn.Module):
    """ACTModel's inference forward with tensors in and out (no dict; the VAE encoder is unused
    at inference, the latent is zero)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state, *images):
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        return self.model({OBS_STATE: state, OBS_IMAGES: list(images)})[0]


class _ACTEngine(nn.Module):
    """Drop-in for ACTPolicy.model: same call and return signature, TensorRT inside."""

    def __init__(self, runner, original):
        super().__init__()
        self.runner = runner
        self.original = original          # used if anything ever calls it in training mode

    def forward(self, batch):
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        if self.training:
            return self.original(batch)
        actions = self.runner(batch[OBS_STATE].float().contiguous(),
                              *[im.float().contiguous() for im in batch[OBS_IMAGES]])
        return actions.float(), (None, None)


def _accelerate_act(policy, precision="fp16"):
    cfg = policy.config
    if cfg.env_state_feature:
        raise NotImplementedError("ACT with an environment-state input is not covered")
    keys = list(cfg.image_features)
    state = torch.zeros(1, cfg.robot_state_feature.shape[0], device="cuda")
    images = [torch.zeros(1, *cfg.image_features[k].shape, device="cuda") for k in keys]
    core = _ACTCore(policy.model).eval()
    key = f"act_{precision}_" + weights_fingerprint(core, (keys, precision, [tuple(i.shape) for i in images]))
    runner, how = build_engine(core, [state, *images], ["state"] + [f"image{i}" for i in range(len(keys))],
                               ["actions"], key, strongly_typed=precision == "fp32",
                               fp16=precision == "fp16")
    policy.model = _ACTEngine(runner, core.model).eval()
    return f"act: whole network as one {precision} TensorRT engine ({how})"


# --------------------------------------------------------------------------------------------
# MolmoAct2
# --------------------------------------------------------------------------------------------
class _MolmoFlowStep(nn.Module):
    """One flow-matching step of MolmoAct2's action expert as a pure tensor function:

        velocity   = expert(trajectory, t | cross-attention to the VLM's per-layer KV)
        trajectory = keep * (trajectory + dt * keep * velocity)

    i.e. one iteration of _run_action_flow_loop (no self mask, no valid_action, not causal; the
    runtime checks this on every call and falls back to PyTorch otherwise). Equal to the original
    in PyTorch: max |diff| 0.0039 over the 10 steps (bf16 rounding; the fused-qkv original vs the
    split version below).
    """

    def __init__(self, expert, rope_cache, dt):
        super().__init__()
        self.expert = expert
        self.register_buffer("rope_cos", rope_cache[0].clone(), persistent=False)
        self.register_buffer("rope_sin", rope_cache[1].clone(), persistent=False)
        self.dt = float(dt)
        # Same weights as each block's fused qkv projection, split into q/k/v projections (the
        # fused projection + view(.., 3, heads, hd) + slicing trips a TensorRT Myelin check).
        self.q, self.k, self.v = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for block in expert.blocks:
            qkv = block.self_attn.qkv
            ws = qkv.weight.chunk(3, dim=0)
            bs = qkv.bias.chunk(3, dim=0) if qkv.bias is not None else (None, None, None)
            for dst, w, b in zip((self.q, self.k, self.v), ws, bs):
                lin = nn.Linear(qkv.in_features, w.shape[0], bias=b is not None, device=w.device, dtype=w.dtype)
                lin.weight.data.copy_(w)
                if b is not None:
                    lin.bias.data.copy_(b)
                dst.append(lin)

    def _self_attn(self, i, sa, x, rope):
        import torch.nn.functional as F

        b, n, _ = x.shape
        q = self.q[i](x).view(b, n, sa.num_heads, sa.head_dim).transpose(1, 2)
        k = self.k[i](x).view(b, n, sa.num_heads, sa.head_dim).transpose(1, 2)
        v = self.v[i](x).view(b, n, sa.num_heads, sa.head_dim).transpose(1, 2)
        q, k = sa._apply_qk_norm(q, k)
        q, k = sa.rope(q, k, rope_cache=rope)
        out = F.scaled_dot_product_attention(q, k, v)
        return sa.out_proj(out.transpose(1, 2).reshape(b, n, sa.hidden_size))

    @staticmethod
    def _cross_attn(ca, x, k, v, mask):
        # Written out (softmax(q k^T / sqrt(d) + mask) v) with scores and softmax in fp32, as the
        # fused bf16 kernels accumulate. kv arrive as (b, S, heads, hd), k already normalized.
        b, n, _ = x.shape
        q = ca.q_proj(x).view(b, n, ca.num_heads, ca.head_dim).transpose(1, 2)
        if ca.q_norm is not None:
            q = ca.q_norm(q)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        w = (q.float() @ k.float().transpose(-1, -2)) * (ca.head_dim ** -0.5) + mask.float()
        out = (w.softmax(dim=-1) @ v.float()).to(x.dtype)
        return ca.out_proj(out.transpose(1, 2).reshape(b, n, ca.hidden_size))

    def forward(self, traj, t, keep, cross_mask, *kv):
        from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import _modulate

        ex = self.expert
        cond = ex._time_conditioning(t)
        x = ex.action_embed(traj)
        rope = (self.rope_cos, self.rope_sin)
        for i, blk in enumerate(ex.blocks):     # ActionExpertBlock.forward
            sh_sa, sc_sa, g_sa, sh_ca, sc_ca, g_ca, sh_ff, sc_ff, g_ff = blk.modulation(cond).chunk(9, dim=1)
            x = x + g_sa.unsqueeze(1) * self._self_attn(i, blk.self_attn, _modulate(blk.self_norm(x), sh_sa, sc_sa), rope)
            x = x + g_ca.unsqueeze(1) * self._cross_attn(blk.cross_attn, _modulate(blk.cross_norm(x), sh_ca, sc_ca),
                                                         kv[2 * i], kv[2 * i + 1], cross_mask)
            x = x + g_ff.unsqueeze(1) * blk.mlp(_modulate(blk.ff_norm(x), sh_ff, sc_ff))
        vel = ex.final_layer(x, cond) * keep
        return (traj + self.dt * vel) * keep


class _MolmoFlowEngine:
    """Replacement for the backbone's _run_action_flow_loop(inputs, steps)."""

    def __init__(self, backbone, runner, steps, horizon):
        self.backbone = backbone
        self.runner = runner
        self.steps = steps
        self.horizon = horizon
        self.fallback = backbone._run_action_flow_loop
        self.calls = 0

    def usable(self, inputs, steps):
        c = inputs.context
        return (steps == self.steps and c.self_mask is None and c.valid_action is None
                and c.cross_mask is not None and inputs.trajectory.shape[0] == 1
                and inputs.trajectory.shape[1] == self.horizon
                and not self.backbone.action_expert.training)

    def __call__(self, inputs, steps):
        if not self.usable(inputs, steps):
            return self.fallback(inputs, steps)
        self.calls += 1
        traj = inputs.trajectory.contiguous()
        valid = self.backbone._action_dim_valid_mask(traj, inputs.action_dim_is_pad)
        keep = (torch.ones_like(traj) if valid is None else valid.to(traj.dtype).expand_as(traj)).contiguous()
        kv = [t.contiguous() for pair in inputs.context.kv_contexts for t in pair]
        cross_mask = inputs.context.cross_mask.contiguous()
        for idx in range(steps):
            t = torch.full((1,), idx / steps, device=traj.device, dtype=torch.float32)
            traj = self.runner(traj, t, keep, cross_mask, *kv)
        return traj


def _accelerate_molmoact2(policy):
    bb = policy._backbone()
    ex = bb.action_expert
    if ex.config.causal_attn:
        raise NotImplementedError("causal action expert not covered")
    steps = int(getattr(policy.config, "num_inference_steps", None) or bb.config.flow_matching_num_steps)
    horizon = bb._resolve_action_horizon(policy._generation_action_horizon())
    dt = ex.action_embed.weight.dtype
    heads, hd = ex.config.num_heads, ex.hidden_size // ex.config.num_heads
    S, n_kv = 290, 2 * len(ex.blocks)               # example context length; the engine takes 16..2048
    rope = ex.blocks[0].self_attn.rope.build_cache(seq_len=horizon, device="cuda", dtype=dt)
    step = _MolmoFlowStep(ex, rope, 1.0 / steps).eval()
    traj = torch.randn(1, horizon, bb.config.max_action_dim, device="cuda", dtype=dt)
    args = [traj, torch.zeros(1, device="cuda"), torch.ones_like(traj),
            torch.zeros(1, 1, 1, S, device="cuda", dtype=dt),
            *[torch.randn(1, S, heads, hd, device="cuda", dtype=dt) for _ in range(n_kv)]]
    names = ["traj", "t", "keep", "cross_mask"] + [f"kv{i}" for i in range(n_kv)]
    s_dim = torch.export.Dim("ctx", min=16, max=2048)
    dyn = (None, None, None, {3: s_dim}, tuple({1: s_dim} for _ in range(n_kv)))
    prof = {"cross_mask": ((1, 1, 1, 16), (1, 1, 1, S), (1, 1, 1, 2048))}
    prof.update({f"kv{i}": ((1, 16, heads, hd), (1, S, heads, hd), (1, 2048, heads, hd)) for i in range(n_kv)})
    key = "molmoact2_flowstep_" + weights_fingerprint(step, (steps, horizon, str(dt)))
    runner, how = build_engine(step, args, names, ["traj_out"], key, dynamic_shapes=dyn, profile=prof)
    bb.action_cuda_graph_manager.set_enabled(False)   # the engine replaces the graphed loop
    eng = _MolmoFlowEngine(bb, runner, steps, horizon)
    bb._run_action_flow_loop = eng
    return f"molmoact2: action expert ({steps} flow steps x {len(ex.blocks)} blocks) as a bf16 TensorRT engine ({how})"


# --------------------------------------------------------------------------------------------
# Per-tick overheads (not TensorRT; same per-tick path, same actions)
# --------------------------------------------------------------------------------------------
def fast_eval(policy):
    """MolmoAct2's select_action calls self.eval() on every tick: a walk over ~1300 modules, ~5 ms
    of CPU per tick. Keep the semantics, skip the walk when already in eval mode."""
    cls_eval = type(policy).eval

    def eval_once():
        return cls_eval(policy) if policy.training else policy

    policy.eval = eval_once


def set_n_action_steps(policy, n):
    """MolmoAct2.__init__ overwrites config.n_action_steps with the base checkpoint's norm-tag
    metadata (30 for SO-100/101), so --policy.n_action_steps never reaches it. Re-apply."""
    n = int(n)
    if not 1 <= n <= policy.config.chunk_size:
        raise ValueError(f"n_action_steps {n} outside 1..{policy.config.chunk_size}")
    policy.config.n_action_steps = n
    policy._action_queue = deque(maxlen=n)


def accelerate(policy, *, act_precision="fp16"):
    target = policy.base_model.model if hasattr(policy, "base_model") else policy
    kind = target.config.type
    if kind == "act":
        return _accelerate_act(target, act_precision)
    if kind == "molmoact2":
        info = _accelerate_molmoact2(target)
        fast_eval(target)
        return info + "; per-tick eval() made O(1)"
    raise NotImplementedError(f"no TensorRT path for policy type {kind!r}")
