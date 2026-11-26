"""Compare trained checkpoints on the same validation images with color-space-independent metrics.

    python3 evaluate.py --out eval/ \
        "1 baseline (notebook)=ckpts/baseline.pt@train" \
        "2 + eval()=ckpts/baseline.pt" \
        "3 + attention fix=ckpts/attnfix.pt" \
        "4 + Lab=ckpts/lab.pt"

Each model runs on one image at a time, as in real use. `@train` reproduces the original
notebook, which never called model.eval() (BatchNorm uses the statistics of that single image).

Metrics (all on RGB reconstructed with the ground-truth V or L channel):
  native_mse   MSE in the model's own target space (HSV hs or scaled Lab ab); not comparable across spaces
  rgb_psnr     PSNR of the RGB image, dB (higher is better)
  ab_rmse      RMSE of the CIE Lab a,b channels (color error, lower is better)
  colorfulness mean Hasler-Suesstrunk colorfulness of the predictions / of ground truth (1.0 = as vivid)
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.color import rgb2lab

from model import UNetModel
from dataset import get_dataset, to_rgb


def colorfulness(rgb):
    r, g, b = [rgb[..., i].astype(np.float64) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)


def evaluate(model, val, color_space, train_mode, limit=None, keep=()):
    model.train(train_mode)
    n = min(limit or len(val), len(val))
    mse, psnr, ab_se, cf_pred, cf_gt, kept = [], [], [], [], [], {}
    with torch.no_grad():
        for i in range(n):
            inp, target = val[i]
            pred = model(inp[None])[0]
            mse.append(F.mse_loss(pred, target).item())
            p_rgb, g_rgb = to_rgb(inp, pred, color_space), to_rgb(inp, target, color_space)
            err = (p_rgb.astype(np.float64) - g_rgb) / 255.0
            psnr.append(10 * np.log10(1.0 / max((err ** 2).mean(), 1e-10)))
            ab_se.append(((rgb2lab(p_rgb)[..., 1:] - rgb2lab(g_rgb)[..., 1:]) ** 2).mean())
            cf_pred.append(colorfulness(p_rgb))
            cf_gt.append(colorfulness(g_rgb))
            if i in keep:
                kept[i] = (p_rgb, g_rgb, inp)
    return {
        "native_mse": float(np.mean(mse)),
        "rgb_psnr": float(np.mean(psnr)),
        "ab_rmse": float(np.sqrt(np.mean(ab_se))),
        "colorfulness": float(np.mean(cf_pred) / np.mean(cf_gt)),
        "n": n,
    }, kept


def label(img, text, h=18):
    canvas = Image.new("RGB", (img.width, img.height + h), "white")
    canvas.paste(img, (0, h))
    ImageDraw.Draw(canvas).text((4, 3), text, fill="black")
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="NAME=CKPT[@train]")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--grid", default="0,3,4,6,9,12,15,21,24,30", help="val indices to show")
    ap.add_argument("--scale", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    idx = [int(i) for i in a.grid.split(",")]
    datasets, results, rows, gray_row, gt_row = {}, [], [], None, None
    for spec in a.models:
        name, _, path = spec.partition("=")
        path, _, mode = path.partition("@")
        model = UNetModel.load_from_checkpoint(path, map_location="cpu", weights_only=False)
        cs = model.hparams.get("color_space", "hsv")
        if cs not in datasets:
            datasets[cs] = get_dataset(cs, os.environ.get("DATA_DIR"))[1]
        m, kept = evaluate(model, datasets[cs], cs, mode == "train", a.limit, keep=set(idx))
        m.update(name=name, ckpt=path, mode=mode or "eval", color_space=cs, attn_fix=model.hparams.get("attn_fix"))
        print(json.dumps(m), flush=True)
        results.append(m)
        rows.append((name, np.concatenate([kept[i][0] for i in idx], axis=1)))
        if gt_row is None:
            gt_row = np.concatenate([kept[i][1] for i in idx], axis=1)
            gray_row = np.concatenate([np.repeat(((kept[i][2][0].numpy() + 0.5).clip(0, 1) * 255)
                                                 .astype(np.uint8)[..., None], 3, axis=-1) for i in idx], axis=1)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    def up(arr):
        im = Image.fromarray(arr)
        return im.resize((im.width * a.scale, im.height * a.scale), Image.NEAREST)

    panels = [label(up(gray_row), "input (grayscale)")]
    panels += [label(up(r), f"{n}   PSNR {m['rgb_psnr']:.2f} dB  ab RMSE {m['ab_rmse']:.2f}  colorfulness {m['colorfulness']:.2f}")
               for (n, r), m in zip(rows, results)]
    panels.append(label(up(gt_row), "ground truth"))
    grid = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)), "white")
    y = 0
    for p in panels:
        grid.paste(p, (0, y))
        y += p.height
    grid.save(os.path.join(a.out, "comparison.png"))
    print("wrote", os.path.join(a.out, "comparison.png"))


if __name__ == "__main__":
    main()
