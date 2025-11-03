import math

import lightning as L
import numpy as np
import torch
import torchmetrics
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn, Tensor
from torchmetrics import MetricCollection

from exo_finder.compute.lc_utils import normalize_flux_minmax
from exo_finder.training.base.focal_loss import FocalLoss
from exo_finder.training.blocks.convolutional_encoder_1d import ConvolutionalEncoder1D, ConvolutionalEncoderParams
from exo_finder.training.blocks.utils_module import DebugModule, UnsqueezeDim


class CleanSegmentationModelConv(L.LightningModule):
    def __init__(self, input_size: int):
        super().__init__()
        self.lr = 1e-3
        self.wd = 1e-2
        self.loss_fn = FocalLoss()
        self._val_metrics = MetricCollection(
            {
                # True positive rate, proportion of all actual positives that were classified correctly as positives
                "recall": torchmetrics.classification.BinaryRecall(),
                # True positive over everything classified as positive
                "precision": torchmetrics.classification.BinaryPrecision(),
                # Harmonic mean of precision and recall
                "binary_f1": torchmetrics.classification.BinaryF1Score(),
                # The Precision-Recall AUC
                "bin_avg_precision": torchmetrics.classification.BinaryAveragePrecision(),
            },
            prefix="val_",
        )

        self._test_metrics = self._val_metrics.clone(prefix="test_")

        final_power = 5
        num_layers = int(math.log2(input_size) - final_power)
        self.encoder = nn.Sequential(
            DebugModule("Before unsqueeze"),
            UnsqueezeDim(dim=2),
            DebugModule("After unsqueeze"),
            ConvolutionalEncoder1D(
                params=ConvolutionalEncoderParams(
                    num_layers=num_layers,
                    kernel_size=15,  # 30 minutes
                    in_features=1,
                )
            ),
        )

        self.decoder = nn.Linear(in_features=2**final_power, out_features=input_size)

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        DebugModule("after encoder").forward(x)
        return self.decoder(x)

    def training_step(self, batched_data, batch_idx):
        x, y = self.get_xy(batched_data)
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log(name="train_loss", value=loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batched_data: dict[str, Tensor]):
        x, y = self.get_xy(batched_data)
        self._val_metrics.update(self(x), y.int())

    def on_validation_epoch_end(self):
        self.log_dict(self._val_metrics.compute())
        self._val_metrics.reset()

    def test_step(self, batched_data: dict[str, Tensor]):
        x, y = self.get_xy(batched_data)
        self._test_metrics.update(self(x), y.int())

    def on_test_epoch_end(self):
        self.log_dict(self._test_metrics.compute())
        self._test_metrics.reset()

    def predict_step(self, batched_data: dict[str, Tensor], threshold: float = 0.5) -> Tensor:
        x, _ = self.get_xy(batched_data)
        return self.predict(x)

    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        return torch.sigmoid(self(x)) > threshold

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self._configure_optimizers_cosine_annealing()

    def _configure_optimizers_cosine_annealing(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def _configure_optimizers_onecycle(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

        # Lightning 2.x has a robust estimate for total training steps
        total_steps = self.trainer.estimated_stepping_batches

        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=3e-3,  # try 3e-3 to 1e-2 for this tiny net. If loss is smooth and falling fast, try 5e-3, 1e-2.
            total_steps=total_steps,  # Lightning computed
            pct_start=0.1,  # 10% warmup
            anneal_strategy="cos",
            div_factor=25.0,  # initial lr = max_lr / div_factor
            final_div_factor=100.0,  # final lr = max_lr / (div_factor*final_div_factor)
            three_phase=False,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",  # OneCycle is step-based
                "frequency": 1,
            },
        }

    def _configure_optimizers_warmup(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(50, int(0.1 * total_steps))  # ~10% warmup

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            # cosine from 1.0 -> ~0 over remaining steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def _configure_optimizers_cosine_annealing_warm_restarts(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        # Restart every epoch, then double the period each time
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.trainer.estimated_stepping_batches, T_mult=2, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def _configure_optimizers_cyclic_triangular2(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        steps_per_epoch = self.trainer.estimated_stepping_batches
        sched = torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=3e-4,
            max_lr=3e-3,
            step_size_up=steps_per_epoch // 2,  # up half epoch, down half epoch
            mode="triangular2",
            cycle_momentum=False,  # AdamW: disable momentum cycling
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    @staticmethod
    def get_xy(batched_data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x = batched_data["syn_lc_data"]
        y = (x < 0).to(torch.float32)
        x = normalize_flux_minmax(x)
        return x, y
