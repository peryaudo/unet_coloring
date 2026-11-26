"""Train the colorizer.

    python3 train.py                                  # original setup (HSV, original attention), wandb
    python3 train.py --config configs/baseline.yaml   # experiment config (used on RunPod)

On RunPod (RUN_ID set by scripts/pod_entry.sh) metrics go to $RUN_DIR/metrics.jsonl, and
checkpoints and sample grids are uploaded to R2 with tools/r2.py.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pL
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from PIL import Image

from model import UNetModel
from dataset import get_dataset, to_rgb

DEFAULTS = {
    "color_space": "hsv",     # hsv | lab
    "attn_fix": False,
    "max_epochs": 200,
    "batch_size": 512,
    "learning_rate": 3e-4,
    "num_workers": 7,
    "seed": 0,
    "ckpt_every_epochs": 25,
    "sample_every_epochs": 25,
    "accelerator": "auto",    # set to gpu on RunPod so a missing GPU fails fast instead of training on CPU
}


def load_experiment(path):
    if not path:
        return dict(DEFAULTS)
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return {**DEFAULTS, **(cfg.get("experiment") or {})}


def sample_grid(model, val_dataset, color_space, n=8):
    """Rows: grayscale input, prediction, ground truth. One image at a time (as in real use), eval mode."""
    was_training = model.training
    model.eval()
    rows = [[], [], []]
    with torch.no_grad():
        for i in range(min(n, len(val_dataset))):
            inp, target = val_dataset[i]
            pred = model(inp[None].to(model.device))[0].cpu()
            gray = ((inp[0].numpy() + 0.5).clip(0, 1) * 255).astype(np.uint8)
            rows[0].append(np.stack([gray] * 3, axis=-1))
            rows[1].append(to_rgb(inp, pred, color_space))
            rows[2].append(to_rgb(inp, target, color_space))
    model.train(was_training)
    grid = np.concatenate([np.concatenate(r, axis=1) for r in rows], axis=0)
    return Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST)


class RunArtifacts(Callback):
    """metrics.jsonl every epoch; checkpoints and sample grids to R2 when running as a RunPod job."""

    def __init__(self, exp, val_dataset):
        self.exp = exp
        self.val_dataset = val_dataset
        self.run_dir = os.environ.get("RUN_DIR", "run/local")
        os.makedirs(self.run_dir, exist_ok=True)
        self.upload = bool(os.environ.get("RUN_ID")) and os.environ.get("R2_UPLOAD", "1") != "0"
        self.best = float("inf")
        self.t0 = time.time()

    def _r2(self):
        from tools import r2
        return r2

    def on_train_epoch_end(self, trainer, pl_module):
        # Runs after the epoch's validation, so both train_loss_epoch and val_loss are available.
        m = {k: float(v) for k, v in trainer.callback_metrics.items()}
        epoch = trainer.current_epoch + 1
        row = {
            "step": trainer.global_step,
            "epoch": epoch,
            "train_loss": m.get("train_loss_epoch"),
            "val_loss": m.get("val_loss"),
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
            "elapsed_sec": round(time.time() - self.t0, 1),
        }
        with open(os.path.join(self.run_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        last = epoch == trainer.max_epochs
        step = trainer.global_step
        if epoch % self.exp["sample_every_epochs"] == 0 or last:
            path = os.path.join(self.run_dir, f"sample_{step:08d}.png")
            sample_grid(pl_module, self.val_dataset, self.exp["color_space"]).save(path)
            if self.upload:
                self._r2().put_sample(path, step=step)
        if epoch % self.exp["ckpt_every_epochs"] == 0 or last:
            path = os.path.join(self.run_dir, "ckpt.pt")
            trainer.save_checkpoint(path)
            if self.upload:
                self._r2().put_ckpt(path, step=step)
            val = row["val_loss"]
            if val is not None and val < self.best:
                self.best = val
                if self.upload:
                    self._r2().put_ckpt(path, best=True)
            os.remove(path)
            if self.upload:
                self._r2().write_status(state="running", step=step, message=f"epoch {epoch} val_loss {val:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: few images, 2 epochs")
    args = ap.parse_args()
    exp = load_experiment(args.config)
    if args.smoke:
        exp.update(max_epochs=2, batch_size=8, num_workers=0, ckpt_every_epochs=1, sample_every_epochs=1,
                   accelerator="cpu")
    print("experiment:", json.dumps(exp), flush=True)

    pL.seed_everything(exp["seed"], workers=True)
    torch.set_float32_matmul_precision("high")
    data_dir = os.environ.get("DATA_DIR")
    train_dataset, val_dataset = get_dataset(exp["color_space"], data_dir, limit=32 if args.smoke else None)
    workers = min(exp["num_workers"], os.cpu_count() or 1)
    train_dataloader = DataLoader(train_dataset, batch_size=exp["batch_size"], shuffle=True, num_workers=workers,
                                  persistent_workers=workers > 0)
    val_dataloader = DataLoader(val_dataset, batch_size=exp["batch_size"], num_workers=workers)

    model = UNetModel(attn_fix=exp["attn_fix"], color_space=exp["color_space"], learning_rate=exp["learning_rate"])

    callbacks = [LearningRateMonitor(logging_interval='step')]
    if args.config:
        callbacks.append(RunArtifacts(exp, val_dataset))
    use_wandb = not args.config and not args.smoke
    if use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project="unet-coloring", log_model=True)
    else:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(os.environ.get("RUN_DIR", "run/local"), name="lightning")
    trainer = pL.Trainer(logger=logger, max_epochs=exp["max_epochs"], callbacks=callbacks,
                         accelerator=exp["accelerator"], enable_checkpointing=not args.config,
                         enable_progress_bar=False, log_every_n_steps=5)
    trainer.fit(model, train_dataloader, val_dataloader)
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
