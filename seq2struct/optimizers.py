import attr
import torch

from seq2struct.utils import registry

registry.register('optimizer', 'adadelta')(torch.optim.Adadelta)
registry.register('optimizer', 'adam')(torch.optim.Adam)
registry.register('optimizer', 'sgd')(torch.optim.SGD)


@registry.register('lr_scheduler', 'warmup_polynomial')
@attr.s
class WarmupPolynomialLRScheduler:
    optimizer = attr.ib()
    num_warmup_steps = attr.ib()
    start_lr = attr.ib()
    end_lr = attr.ib()
    decay_steps = attr.ib()
    power = attr.ib()

    def update_lr(self, current_step):
        if current_step < num_warmup_steps:
            warmup_frac_done = current_step / self.num_warmup_steps
            new_lr = self.start_lr * warmup_frac_done
        else:
            new_lr = (
                (self.start_lr - self.end_lr) * (1 - current_step / self.decay_steps) ** self.power
                + self.end_lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


@registry.register('lr_scheduler', 'noop')
class NoOpLRScheduler:
    def __init__(self, optimizer):
        pass

    def update_lr(self, current_step):
        pass
