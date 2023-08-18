# CycleGAN

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Inference](#inference)
- [Training](#training)
- [Acknowledgements](#acknowledgements)

## Results
![1](https://github.com/lukabarbakadze/CycleGAN/blob/main/test_imgs/results.png)
![2](https://github.com/lukabarbakadze/CycleGAN/blob/main/test_imgs/monet.png)

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/lukabarbakadze/CycleGAN.git
   ```

2. Create and activate a virtual environment (e.g., using conda):
   ```sh
   conda create -n myenv python==3.11
   conda activate myenv
   ```

3. Navigate to the CycleGAN directory and install dependencies:
   ```sh
   cd CycleGAN
   pip install -r requirements.txt
   ```

## Dataset Preparation
Arrange your dataset as follows:
   ```sh
   CycleGAN/
   ├── data/
   │   ├── monet_jpg/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   ├── other_jpg/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   ```

## Inference

An inference script is provided in inference.ipynb file.

## Training
1. Configure desired parameters in config/config.py and save changes.

2. Run the training script:
   ```sh
   python scripts/train.py
   ```

3. Track training logs using TensorBoard:
   ```sh
   tensorboard --logdir=tb_logs/lightning_logs
   ```

## Acknowledgements
* [Paper: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
* [CycleGAN implementation from scratch by Aladdin Persson](https://www.youtube.com/watch?v=4LktBHGCNfw&t=1369s)
* [Dataset](https://www.kaggle.com/competitions/gan-getting-started/data)