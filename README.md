# 🎮 Minecraft Skin Generator (DCGAN)

## About

Minecraft Skin Generator is a deep learning project that uses a Deep Convolutional Generative Adversarial Network (DCGAN) to produce original Minecraft skins. Trained on a dataset of player-created skins, this tool generates coherent and stylized 64×64 pixel skins in classic Minecraft format. A Streamlit-based web app is included for interactive use and real-time skin generation.

This project combines an end-to-end GAN pipeline (via `SkinGen.py`) and a web interface (via `Skinapp.py`) for creators, modders, and players who want to design custom skins or experiment with generative models.

## Features

- **🎨 AI-Generated Skins**  
  Generates full-body Minecraft skins (64×64) using a learned DCGAN model.

- **🧠 End-to-End Training Pipeline**  
  `SkinGen.py` handles dataset loading, preprocessing, model definition, training, and skin generation—all in one script.

- **📦 Pretrained Weights Included**  
  `generator.h5` allows users to immediately generate skins without retraining.

- **🖼️ Visualization Support**  
  Sample outputs (e.g., `epoch_050.png`) are saved during training to visualize model progress.

- **🖥️ Streamlit Web App**  
  `Skinapp.py` provides a browser interface to generate new skins in real time using the trained model.

- **🧪 Customization Friendly**  
  Modify architecture, batch size, learning rate, or replace DCGAN with other GAN variants easily within `SkinGen.py`.

## File Overview

| File             | Description                                              |
|------------------|----------------------------------------------------------|
| `SkinGen.py`     | Full DCGAN pipeline: data loading, training, generation  |
| `Skinapp.py`     | Streamlit interface for real-time skin generation        |
| `generator.h5`   | Pretrained generator model for inference                 |
| `epoch_050.png`  | Example output at epoch 50                               |
| `README.md`      | Project documentation                                    |
| `LICENSE`        | Open-source license (MIT)                                |

---

## Technologies Used

- **Deep Learning**: Keras (TensorFlow backend), DCGAN  
- **Data Processing**: Python, NumPy, PIL  
- **Visualization**: Matplotlib  
- **Frontend**: Streamlit (real-time web interface)  
- **Model Storage**: HDF5 (`generator.h5`)

## Future Improvements

- Add style interpolation sliders to Streamlit UI  
- Expand dataset diversity with fantasy or sci-fi themed skins  
- Implement conditional GANs for labeled/class-specific generation  
- Export `.mcpack` or `.zip` formats for direct Minecraft import  
- Deploy as a hosted web app for public access

## Example Usage

### 1. Clone the Repository

```bash
git clone https://github.com/neelmajmudar/MinecraftSkinGen.git
cd MinecraftSkinGen

pip install -r requirements.txt

streamlit run Skinapp.py
```

