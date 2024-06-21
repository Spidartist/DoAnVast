import math

def get_warmup_cosine_lr(base_lr, max_lr, cur, total_steps, steps_per_epoch, warmup_epochs=2):
    """ warmup in first 2 epochs, then lr is calculated using cosine function
    """

    if cur <= warmup_epochs * steps_per_epoch:
        lr = base_lr + cur * (max_lr - base_lr)/(warmup_epochs*steps_per_epoch)
    else:
        step = cur - warmup_epochs * steps_per_epoch
        decayed_steps = total_steps - warmup_epochs * steps_per_epoch
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decayed_steps))
        lr = max_lr * cosine_decay
    
    return max(0., lr)

class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for idx, group in enumerate(self.optimizer.param_groups):
            if idx == 0:
                group['lr'] = new_lr * 0.1
                # group['lr'] = new_lr
            elif idx == 1:
                group['lr'] = new_lr

        return new_lr