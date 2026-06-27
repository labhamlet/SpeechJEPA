import math
from torch.optim.lr_scheduler import LambdaLR


def get_tri_stage_schedule(
    optimizer,
    total_steps,
    phase_ratio=(0.1, 0.4, 0.5),
    init_lr_scale=0.01,
    final_lr_scale=0.01,
):
    """Tri-stage LR schedule (https://arxiv.org/abs/1904.08779), matching
    fairseq's `tri_stage`. The optimizer's base LR is treated as the peak LR.

      - warmup: rises linearly from init_lr_scale*peak -> peak
      - hold:   stays at peak
      - decay:  decays *exponentially* from peak -> final_lr_scale*peak
      - after:  held at final_lr_scale*peak

    LambdaLR multiplies the base (peak) LR by the factor returned below.
    """
    assert abs(sum(phase_ratio) - 1.0) < 1e-8, "phase ratios must sum to 1.0"

    warmup_steps = int(phase_ratio[0] * total_steps)
    hold_steps = int(phase_ratio[1] * total_steps)
    decay_steps = total_steps - warmup_steps - hold_steps  # absorbs rounding

    warmup_rate = (1.0 - init_lr_scale) / warmup_steps if warmup_steps > 0 else 0.0
    decay_factor = -math.log(final_lr_scale) / decay_steps if decay_steps > 0 else 0.0

    def lr_lambda(current_step):
        if current_step < warmup_steps:                         # warmup
            return init_lr_scale + warmup_rate * current_step
        if current_step < warmup_steps + hold_steps:            # hold
            return 1.0
        step_in_decay = current_step - warmup_steps - hold_steps
        if step_in_decay <= decay_steps:                        # exponential decay
            return math.exp(-decay_factor * step_in_decay)
        return final_lr_scale                                   # constant tail

    return LambdaLR(optimizer, lr_lambda)