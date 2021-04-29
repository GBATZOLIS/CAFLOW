from pytorch_lightning.callbacks import Callback

class EMACallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        pl_module.ema.store(pl_module.parameters())
        pl_module.ema.copy_to(pl_module.parameters())

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.ema.restore(pl_module.parameters())