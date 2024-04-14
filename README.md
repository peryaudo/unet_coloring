# UNet based image colorizer

![](example.png)

## What is this?

It gives color to grayscale flower images using U-Net with self-attention. [Oxford 102 Flower Dataset](https://paperswithcode.com/dataset/oxford-102-flower) is used.


## Usage

If you'd like using vast.ai, a 24GB RTX4090 or RTX3090 instance is recommended.

Preparation:

```
$ pip install -r requirements.txt
$ wandb login
```

To generate the dataset split (Oxford 102 Flower does not have train/val split):

```
$ python3 generate_split.py
```

Then run the training:

```
$ python3 train.py
```

[inference.ipynb](inference.ipynb) shows how to run the inference from a trained checkpoint.

## TODO

- [ ] Try loss function other than MSE
- [ ] Try datasets other than Oxford 102 Flower
- [ ] Try GAN or diffusion based approach

## License

MIT