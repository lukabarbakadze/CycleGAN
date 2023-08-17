from tqdm import tqdm
from torchvision.utils import save_image
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model.discriminator import Discriminator
from model.generator import Generator
from dataset.dataset import CycleGanDataset
from model.CycleGAN import CycleGAN_LightningSystem
import config.config as config


def build_model():
    disc_M = Discriminator(
        in_channels=config.IN_CHANNELS, 
        features=config.FEATURES
    ).to(config.DEVICE)

    disc_O = Discriminator(
        in_channels=config.IN_CHANNELS, 
        features=config.FEATURES
    ).to(config.DEVICE)

    gen_M = Generator(
        img_channels=config.IN_CHANNELS, 
        num_features=config.NUM_FEATURES, 
        num_residuals=config.NUM_RESIDUALS
    ).to(config.DEVICE)

    gen_O = Generator(
        img_channels=config.IN_CHANNELS, 
        num_features=config.NUM_FEATURES, 
        num_residuals=config.NUM_RESIDUALS
    ).to(config.DEVICE)

    loader = CycleGanDataset(
        data_dir="data", 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS, 
        transform=config.TRANSFORMS)

    model = CycleGAN_LightningSystem(
        disc_M=disc_M,
        disc_O= disc_O,
        gen_M=gen_M,
        gen_O=gen_O,
        loader=loader
    )
    return model, loader

def train():
    model, loader = build_model()
    logger = TensorBoardLogger(
        "tb_logs",
    )
    trainer = pl.Trainer(
        min_epochs=1, 
        max_epochs=config.NUM_EPOCHS,
        logger=logger,
        accelerator=config.DEVICE,
        default_root_dir="/kaggle/working/"
    )
    
    trainer.fit(model, loader)
    trainer.save_checkpoint("/kaggle/working/last_checkpoint.ckpt")

if __name__=="__main__":
    train()