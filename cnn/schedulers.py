"""Learning rate schedulers."""
import math
from cnn.core import xp


class CosineLR:
    """Cosine learning rate scheduler with warmup (step-based)."""

    def __init__(self, optimizer, max_lr, min_lr, total_steps, warmup_steps=100):
        """
        Args:
            optimizer: optimizer instance (PyTorch or custom)
            max_lr: maximum learning rate after warmup
            min_lr: minimum learning rate at end of decay
            total_steps: total number of training steps (epochs * steps_per_epoch)
            warmup_steps: number of warmup steps (default 100)
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.warmup_steps = max(warmup_steps, 100)
        self.decay_steps = total_steps - self.warmup_steps

        self.current_step = 0

        # Pre-compute schedule
        warmup_lr = xp.linspace(min_lr, max_lr, self.warmup_steps)

        decay_lr = []
        for step in range(1, self.decay_steps + 1):
            alpha = math.cos(math.pi * step / self.decay_steps)
            decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))

        self.schedule = xp.concatenate([warmup_lr, decay_lr])

    def step(self):
        """Update learning rate for current step."""
        if self.current_step < len(self.schedule):
            lr = self.schedule[self.current_step]
        else:
            lr = self.min_lr

        # Update optimizer (works for both PyTorch and custom optimizers)
        if hasattr(self.optimizer, 'param_groups'):
            # PyTorch optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Custom optimizer (has .lr attribute)
            self.optimizer.lr = float(lr)

        self.current_step += 1

    def get_last_lr(self):
        """Get current learning rate."""
        if hasattr(self.optimizer, 'param_groups'):
            return self.optimizer.param_groups[0]['lr']
        else:
            return self.optimizer.lr