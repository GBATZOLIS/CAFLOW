from pytorch_lightning.callbacks import Callback
from caflow.utils.ExponentialMovingAverage import ExponentialMovingAverage

class EMACallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        pl_module.ema = ExponentialMovingAverage(pl_module.parameters(), decay=0.999)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        pl_module.ema.update(pl_module.parameters())

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        pl_module.ema.store(pl_module.parameters())
        pl_module.ema.copy_to(pl_module.parameters())

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.ema.restore(pl_module.parameters())