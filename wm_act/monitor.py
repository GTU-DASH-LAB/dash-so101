"""Runtime physics monitor: compares the world model's expectations against reality.

At every control step, the policy's WM head predicts the visual latent
`wm_future_offset` frames ahead. When that frame arrives, the gap between
expectation and observation ("surprise") is a physics-violation signal: a missed
grasp, a dropped object, or a collision all make reality diverge from what the
model expected its own actions to cause.

Usage inside a control loop:

    monitor = SurpriseMonitor(policy, threshold=...)   # threshold from selfcheck.py
    ...each step:
    surprise = monitor.step(step_idx, obs_batch, action_chunk)
    if monitor.anomalous(surprise):
        # abort the chunk, re-ground, retry (see so_brain/run.py attempt loop)
"""

from torch import Tensor


class SurpriseMonitor:
    def __init__(self, policy, threshold: float):
        self.policy = policy
        self.threshold = threshold
        self.horizon = policy.config.wm_future_offset
        self._expected: dict[int, Tensor] = {}

    def step(self, step_idx: int, batch: dict[str, Tensor], action_chunk: Tensor) -> float | None:
        """Feed the current observation and the action chunk being executed.
        Returns the surprise for this step, or None until the horizon fills."""
        latent = self.policy.visual_latent(batch)
        surprise = None
        if step_idx in self._expected:
            predicted = self._expected.pop(step_idx)
            surprise = float(self.policy.surprise(predicted, latent).mean())
        self._expected[step_idx + self.horizon] = self.policy.predict_future_latent(latent, action_chunk)
        # Drop expectations whose frame was skipped (dropped frames, resets).
        self._expected = {k: v for k, v in self._expected.items() if k > step_idx}
        return surprise

    def anomalous(self, surprise: float | None) -> bool:
        return surprise is not None and surprise > self.threshold

    def reset(self) -> None:
        self._expected.clear()
