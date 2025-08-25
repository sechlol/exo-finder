import lightning as L
import torch
from torch import nn, Tensor

from exo_finder.compute.lc_utils import normalize_flux_minmax
from exo_finder.training.base.loss_functions_segmentation import DiceBCELoss


class CleanSegmentationModel(L.LightningModule):
    def __init__(self, input_size: int, bottleneck_size: int = 32):
        super().__init__()
        self.lr = 3e-4
        self.wd = 1e-2
        self.loss_fn = DiceBCELoss()

        self.net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=bottleneck_size),
            nn.BatchNorm1d(num_features=bottleneck_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=bottleneck_size, out_features=input_size),
        )

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batched_data, batch_idx):
        x, y = self.get_xy(batched_data)
        loss = self.loss_fn(self(x), y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batched_data: dict[str, Tensor]):
        x, y = self.get_xy(batched_data)
        val_loss = self.loss_fn(self(x), y)
        self.log("val_loss", val_loss, prog_bar=True)

    def test_step(self, batched_data: dict[str, Tensor]):
        x, y = self.get_xy(batched_data)
        test_loss = self.loss_fn(self(x), y)
        self.log("test_loss", test_loss)

    def predict_step(self, batched_data: dict[str, Tensor], threshold: float = 0.5) -> Tensor:
        x, _ = self.get_xy(batched_data)
        return self.predict(x)

    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        return torch.sigmoid(self(x)) > threshold

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    @staticmethod
    def get_xy(batched_data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x = batched_data["syn_lc_data"]
        y = (x < 0).to(torch.float32)
        x = normalize_flux_minmax(x)
        return x, y
