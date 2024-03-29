from torch.utils.data import DataLoader
import pytorch_lightning as pL
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb

from model import UNetModel
from dataset import get_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = get_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=7)
    val_dataloader = DataLoader(val_dataset, batch_size=512, num_workers=7)

    model = UNetModel()
    wandb_logger = WandbLogger(project="unet-coloring", log_model=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pL.Trainer(logger=wandb_logger, max_epochs=200, callbacks=[lr_monitor])
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()
