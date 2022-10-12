import sys
import time
import os
import torch.cuda
import numpy as np
import torchvision
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    Callback,
    LearningRateMonitor,
)
from pytorch_lightning.callbacks.progress.tqdm_progress import (
    Tqdm,
    TQDMProgressBar,
)
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info


class SmoothedProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        return Tqdm(
            desc=self.train_description,
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0.75,
        )


class EpochVram(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        root_idx = trainer.strategy.root_device.index
        torch.cuda.reset_peak_memory_stats(root_idx)
        torch.cuda.synchronize(root_idx)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        root_idx = trainer.strategy.root_device.index
        torch.cuda.synchronize(root_idx)
        max_memory = torch.cuda.max_memory_allocated(root_idx) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f'Average Epoch time: {epoch_time:.2f} seconds')
            rank_zero_info(f'Average Peak memory {max_memory:.2f}MiB')
        except AttributeError:
            pass


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        val_batch_frequency=1,
        clamp=True,
        rescale=True,
        disabled=False,
        log_images_kwargs=None,
        log_images_to_loggers=False,
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.disabled = disabled or max_images <= 0
        self.log_images_kwargs = log_images_kwargs or {}
        self.custom_step = 0
        self.custom_val_step = 0
        self.log_images_to_loggers = log_images_to_loggers

    def log_local(
        self, save_dir, split, images, global_step, current_epoch,
    ):
        root = os.path.join(save_dir, 'images', split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = '{}_r-{}_gs-{:06}_e-{:06}.png'.format(
                k, rank_zero_only.rank, global_step, current_epoch,
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, split='train'):
        if not hasattr(pl_module, 'log_images'):
            return

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(
                batch, split=split, **self.log_images_kwargs
            )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            save_dir = None
            for logger in pl_module.loggers:
                if self.log_images_to_loggers and hasattr(logger, 'experiment') and hasattr(
                    logger.experiment, 'add_image'
                ):
                    for k in images:
                        # tensorboard doesn't sort images so prepend tag with epoch and step
                        tag = f'{split}/e{pl_module.current_epoch:04d}/gs{pl_module.global_step:07d}/{k}'
                        if hasattr(logger.experiment, 'add_images'):
                            logger.experiment.add_images(
                                tag,
                                (images[k] + 1.0) / 2.0,
                                global_step=pl_module.global_step,
                            )
                        else:
                            grid = torchvision.utils.make_grid(images[k])
                            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                            logger.experiment.add_image(
                                tag, grid, global_step=pl_module.global_step
                            )
                save_dir = save_dir or getattr(logger, 'save_dir')

            if save_dir:
                self.log_local(
                    save_dir,
                    split,
                    images,
                    pl_module.global_step,
                    pl_module.current_epoch,
                )

        if is_train:
            pl_module.train()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if not self.disabled:
            self.custom_step += 1
            if self.custom_step % self.batch_freq == 0:
                self.log_img(pl_module, batch, split='train')

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not self.disabled:
            self.custom_val_step += 1
            if self.custom_val_step % self.val_batch_freq == 0:
                self.log_img(pl_module, batch, split='val')
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (
                pl_module.calibrate_grad_norm and batch_idx % 25 == 0
            ) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CustomCheckpointing(Callback):
    def __init__(
        self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.logged = False

    # Remove optimizer state from checkpoints
    # but keeping other state
    # (would use save_weights_only but I need to keep fit loop state)
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not self.logged:
            print(f'   !!!! setupcallback ckpt keys\n{checkpoint.keys()}\n\n')
            self.logged = True
        for k in ('optimizer_states', 'lr_schedulers'):
            if k in checkpoint:
                checkpoint[k] = []
        for k in ('callbacks',):
            if k in checkpoint:
                checkpoint[k] = {}

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0 and pl_module.global_step > 5:
            print('on_exception: Summoning checkpoint.')
            try:
                ckpt_path = os.path.join(self.ckptdir, 'on_exception.ckpt')
                trainer.save_checkpoint(ckpt_path)
            except Exception as e:
                print('Error saving on_exception checkpoint', type(e), e)

    # previously on_pretrain_routine_start
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if 'callbacks' in self.lightning_config:
                if (
                    'metrics_over_trainsteps_checkpoint'
                    in self.lightning_config['callbacks']
                ):
                    os.makedirs(
                        os.path.join(self.ckptdir, 'trainstep_checkpoints'),
                        exist_ok=True,
                    )
            print('Project config')
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, '{}-project.yaml'.format(self.now)),
            )

            print('Lightning config')
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({'lightning': self.lightning_config}),
                os.path.join(
                    self.cfgdir, '{}-lightning.yaml'.format(self.now)
                ),
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, 'child_runs', name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass
