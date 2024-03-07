import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pL

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, has_max_pool=True):
        super().__init__()
        self.has_max_pool = has_max_pool
        if self.has_max_pool:
            self.max_pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.has_max_pool:
            x = self.max_pool(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    
    def forward(self, x, resid):
        x = F.relu(self.up_conv(x))
        assert x.shape == resid.shape
        x = torch.cat((resid, x), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_pos):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Embedding(num_pos, embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.register_buffer('pos', torch.arange(0, num_pos))
        
    def forward(self, x):
        x_shape = x.shape
        # [B,C,H,W] -> [B,C,H*W] -> [H*W,B,C]
        x = torch.flatten(x, start_dim=2).permute(2, 0, 1)
        # [H*W,B,C] + [H*W,1,C]
        x = x + self.pos_embed(self.pos.view(-1, 1))
        x = self.ln1(x)
        x, _ = self.self_attn(x, x, x)
        x = self.ln2(x)
        # [H*W,B,C] -> [B,C,H*W]
        x = x.permute(1, 2, 0)
        # [B,C,H*W] -> [B,C,H,W]
        x = x.view(x_shape)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc0 = EncoderBlock(1, 64, has_max_pool=False)
        self.enc1 = EncoderBlock(64, 128)
        self.enc2 = EncoderBlock(128, 256)
        self.enc3 = EncoderBlock(256, 512)
        self.enc4 = EncoderBlock(512, 1024)
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        self.head = nn.Conv2d(64, 2, 1, padding='same')

        self.attn1 = SelfAttentionBlock(1024, num_heads=8, num_pos=16)
        self.attn2 = SelfAttentionBlock(512, num_heads=8, num_pos=64)
        self.attn3 = SelfAttentionBlock(256, num_heads=8, num_pos=256)
        
    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x = self.enc4(x3)
        
        x = self.attn1(x)
        x = self.dec4(x, x3)
        x = self.attn2(x)
        x = self.dec3(x, x2)
        x = self.attn3(x)
        x = self.dec2(x, x1)
        x = self.dec1(x, x0)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x

class UNetModel(pL.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet()
        self.learning_rate = 3e-4
    
    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": optimizer, "lr_scheduler": {'scheduler': lr_scheduler, 'interval': 'step'}}