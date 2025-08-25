import lightning as L
import torch
from lightning.pytorch.callbacks import TQDMProgressBar

import exo_finder.constants as c
from exo_finder.training.base.model_io import get_checkpoint_path
from exo_finder.training.clean_transit_prediction.transit_segmentation_model import CleanSegmentationModel
from exo_finder.training.data_loading.lightcurve_dataset import LcDataModule
from paths import MODEL_DATA_PATH

CHECKPOINT_PATH = MODEL_DATA_PATH / "clean_segmentation_model"


def new_train():
    data_module = LcDataModule(batch_size=64)
    model = CleanSegmentationModel(input_size=c.LC_WINDOW_SIZE, bottleneck_size=32)

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        callbacks=[TQDMProgressBar(refresh_rate=50)],
        default_root_dir=CHECKPOINT_PATH,
        # profiler=AdvancedProfiler(dirpath=".", filename="perf_logs"),
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)


def continue_train():
    data_module = LcDataModule(batch_size=64)
    checkpoint_path = get_checkpoint_path(base_path=CHECKPOINT_PATH, version=2)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model = CleanSegmentationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=checkpoint.get("epoch", 0) + 10,
        accelerator="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        callbacks=[TQDMProgressBar(refresh_rate=50)],
        default_root_dir=CHECKPOINT_PATH,
    )

    # Continue training
    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    new_train()
