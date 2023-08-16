from typing import Any
import torch
import torch.optim as optim
from torchvision.utils import make_grid

import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
import config.config as config

class CycleGAN_LightningSystem(pl.LightningModule):
    def __init__(
            self,
            disc_M,
            disc_O,
            gen_M,
            gen_O,
            loader
        ):
        super().__init__()
        self.disc_M = disc_M # gen_H
        self.disc_O = disc_O # gen_Z
        self.gen_M = gen_M
        self.gen_O = gen_O
        self.loader = loader
        self.l1 = MeanAbsoluteError()
        self.mse = MeanSquaredError()

        self.automatic_optimization = False
    
    def configure_optimizers(self) -> Any:
        self.gen_opt = optim.Adam(
            list(self.gen_M.parameters()) + list(self.gen_O.parameters()),
            lr = config.LEARNING_RATE["G"],
            betas = config.BETAS["G"],
        )
        self.disc_opt = optim.Adam(
            list(self.disc_M.parameters()) + list(self.disc_O.parameters()),
            lr = config.LEARNING_RATE["D"],
            betas = config.BETAS["D"],
        )
        return self.gen_opt, self.disc_opt
    
    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()
        
        # Monet (zebra, Z)
        # Other (horse, H)

        M, O = batch

        ### optimize Disctiminator
        fake_O = self.gen_O(M)
        D_O_real = self.disc_O(O)
        D_O_fake = self.disc_O(fake_O.detach())
        D_O_real_loss = self.mse(D_O_real, torch.ones_like(D_O_real))
        D_O_fake_loss = self.mse(D_O_fake, torch.ones_like(D_O_fake))
        D_O_loss = D_O_real_loss + D_O_fake_loss

        fake_M = self.gen_M(O)
        D_M_real = self.disc_M(M)
        D_M_fake = self.disc_M(fake_M.detach())
        D_M_real_loss = self.mse(D_M_real, torch.ones_like(D_M_real))
        D_M_fake_loss = self.mse(D_M_fake, torch.zeros_like(D_M_fake))
        D_M_loss = D_M_real_loss + D_M_fake_loss

        D_loss = (D_O_loss + D_M_loss) / 2

        gen_opt.zero_grad()
        self.manual_backward(D_loss)
        gen_opt.step()

        ### optimize Generator

        # Monet (zebra, Z)
        # Other (horse, H)

        # adversarial loss for both generators
        D_O_fake = self.disc_O(fake_O)
        D_M_fake = self.disc_M(fake_M)
        loss_G_O = self.mse(D_O_fake, torch.ones_like(D_O_fake))
        loss_G_M = self.mse(D_M_fake, torch.ones_like(D_M_fake))

        # cycle loss
        cycle_monet = self.gen_M(fake_O)
        cycle_other = self.gen_O(fake_M)
        cycle_monet_loss = self.l1(M, cycle_monet)
        cycle_other_loss = self.l1(O, cycle_other)

        # identity loss
        identity_monet = self.gen_M(M)
        identity_other = self.gen_O(O)
        identity_monet_loss = self.l1(M, identity_monet)
        identity_other_loss = self.l1(O, identity_other)

        # full loss
        G_loss = (
            loss_G_M +
            loss_G_O + 
            cycle_monet_loss * config.LAMBDA_CYCLE + 
            cycle_other_loss * config.LAMBDA_CYCLE + 
            identity_monet_loss * config.LAMBDA_IDENTITY +
            identity_other_loss * config.LAMBDA_IDENTITY
        )

        disc_opt.zero_grad()
        self.manual_backward(G_loss)
        disc_opt.step()

        self.log_dict({"g_loss": G_loss, "d_loss": D_loss}, prog_bar=True)

        if (config.LOG_IMAGE == True) and (batch_idx % 50 == 0):
            grid = make_grid(torch.cat((O,fake_M.detach()), dim=0))
            self.logger.experiment.add_image("Real Image & Monet Style Image", grid, self.global_step)
        
    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['gen_M_state_dict'] = self.gen_M.state_dict()

    def on_load_checkpoint(self, checkpoint) -> None:
        if 'gen_M_state_dict' in checkpoint:
            self.gen_M.load_state_dict(checkpoint['gen_M_state_dict'])
    
    def forward(self, x):
        return self.gen_M(x)