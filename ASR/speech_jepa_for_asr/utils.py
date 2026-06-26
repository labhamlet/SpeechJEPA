
import math 
from torch.optim.lr_scheduler import LambdaLR

def get_tri_state_schedule(optimizer, total_steps, phase_ratio=(0.1, 0.4, 0.5),
                           init_lr_scale=0.01, final_lr_scale=0.05):
    warmup_steps = int(phase_ratio[0] * total_steps)
    hold_steps   = int(phase_ratio[1] * total_steps)
    decay_steps  = max(1, total_steps - warmup_steps - hold_steps)
    decay_factor = -math.log(final_lr_scale) / decay_steps   # matches fairseq

    def lr_lambda(step):
        if step < warmup_steps:
            # linear warmup from init_lr_scale
            return init_lr_scale + (1.0 - init_lr_scale) * (step / max(1, warmup_steps))
        elif step < warmup_steps + hold_steps:
            return 1.0
        elif step < total_steps:
            # EXPONENTIAL decay to final_lr_scale
            return math.exp(-decay_factor * (step - warmup_steps - hold_steps))
        else:
            return final_lr_scale

    return LambdaLR(optimizer, lr_lambda)