# 🎮 Minecraft Skin Generator (DCGAN)

## About
Minecraft Skin Generator leverages a Deep Convolutional Generative Adversarial Network (DCGAN) to generate unique Minecraft skins. Trained on a curated dataset of player-created skins, this model produces coherent, pixel-art-style human or character skins ready for in-game use. The goal is to empower players and creators by providing an interactive, creative tool to generate, explore, and customize new skin designs.

DCGANs are a class of GANs that replace fully connected layers with convolutional and transpose-convolutional layers—improving training stability and visual coherence :contentReference[oaicite:1]{index=1}. This project adapts that architecture to the 64×64 skin domain, generating high-quality textures with minimal artifacts.

## Features
- **Generative skin creation**  
  Produces full 64×64 Minecraft skins using learned latent representations.

- **DCGAN architecture**  
  Employs convolutional layers, batch normalization, dropout, and ReLU/LeakyReLU activations following DCGAN best practices :contentReference[oaicite:2]{index=2}.

- **Latent-space exploration**  
  Interactive sampling allows users to adjust noise vectors to generate diverse or interpolated skin outputs.

- **Pretrained model**  
  Comes with pretrained weights—ready to generate new skins instantly, or retrain on new datasets.

- **Modular pipeline**  
  Includes scripts for data prep (`prepare_data.py`), training (`train.py`), generation (`generate.py`), and visualization of outputs across training epochs.

- **Customization ready**  
  Easily adapt learning rates, batch size, loss functions, or try alternative GAN variants like conditional GANs or StyleGAN extensions.

## Project overview
A structured, end-to-end GAN pipeline:

### 1. Data Preparation  
- Input: Folder of raw `.png` skin images, resized/cropped to 64×64.  
- `prepare_data.py` standardizes these images and packages them into a PyTorch `Dataset`.

### 2. Model Architecture  
- **Generator**: Noise → series of transpose-convolution layers → 64×64×3 RGB skin.  
- **Discriminator**: 64×64×3 input → Conv layers → scalar real/fake output.  
- Uses batch norm in all layers (except generator output), ReLU in Generator, LeakyReLU in Discriminator :contentReference[oaicite:3]{index=3}.

### 3. Training Process  
- Implemented in `train.py` with alternating optimization of generator and discriminator using Adam optimizer.  
- Includes checkpoint saving and optional image grid outputs at epochs for training monitoring.

### 4. Skin Generation  
- `generate.py` loads trained model, accepts latent vectors or random noise, outputs new skin PNGs.  
- CLI options for batch size, seed setting, and output directories.

### 5. Visualization & Interpolation  
- Optionally visualize generated samples across epochs (e.g. 10 epochs, 20 epochs...).  
- Support for linear interpolation between two noise vectors to morph between skin designs.

---

## Example Usage

```bash
git clone https://github.com/neelmajmudar/MinecraftSkinGen.git
cd MinecraftSkinGen
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Prepare skin dataset
python prepare_data.py --data_dir ./skins_raw --output data/skins.pt

# Train DCGAN model
python train.py --data data/skins.pt --epochs 50 --batch_size 64 --lr 0.0002 --save_dir models/

# Generate new skins
mkdir -p output
python generate.py --model models/dcgan_skin.pt --num 16 --output output/

# Interpolate between two skins
python generate.py --model models/dcgan_skin.pt --interpolate --steps 8 --output interp/
