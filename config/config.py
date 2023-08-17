import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"
BATCH_SIZE = 1
LEARNING_RATE = {"G": 2e-4, "D": 2e-4}
BETAS = {"G": (0.5, 0.999), "D": (0.5, 0.999)}
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
IN_CHANNELS = 3
NUM_FEATURES = 64
FEATURES = [64, 128, 256, 512]
NUM_RESIDUALS = 9

# for logging image
LOG_IMAGE = True
LOG_IMAGE_DIR = "grid_images"


# LOAD_MODEL = True
# SAVE_MODEL = True
# CHECKPOINT_GEN_H = "genh.pth.tar"
# CHECKPOINT_GEN_H = "genz.pth.tar"
# CHECKPOINT_GEN_H = "critich.pth.tar"
# CHECKPOINT_GEN_H = "criticz.pth.tar"

TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ]
)