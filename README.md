# Insect Acoustic Species Identification (ICT-4442)
### Module: Feature Extraction, Augmentation & Convolutional Baseline (CNN & EfficientNetV2)
**Author**: Saurabh Tiwari (Reg No: 230911238)  
**Branch**: `saurabh`

---

## 🎯 What is in this Branch?

This branch contains the **shared bioacoustic data pipeline** and the **convolutional model family** for our project:
1. **128-Band Log-Mel Spectrogram Extractor**: Standardized bioacoustic time–frequency representation.
2. **Data Augmentation Pipeline**: Waveform-level perturbations (time shift, noise, gain) + **SpecAugment** (frequency and time masking).
3. **PyTorch Dataset & DataLoader**: Supports both real audio files (via metadata CSV) and a built-in synthetic signal generator for quick testing.
4. **Convolutional Models**:
   - `LogMelCNN`: Custom 4-stage VGG-style CNN baseline with auxiliary temperature conditioning.
   - `InsectEfficientNetV2`: Pretrained `EfficientNetV2-S` transfer learning model fine-tuned on spectrograms.
5. **Training & Metrics Engine**: PyTorch training loop with Cosine Annealing, Early Stopping, Top-1 Accuracy, Macro-F1, and Long-Tail Tier analysis.

---

## 👥 How Teammates Can Use This Code

All modules are designed to be modular and easy to import into your respective models.

### 1. For Harsh (Audio Spectrogram Transformer - AST)
You need 128-band log-Mel spectrograms and SpecAugment to feed your transformer:
```python
import torch
from src.data.features import LogMelExtractor
from src.data.augmentation import SpecAugment

# Extract 128-band log-Mel spectrogram from raw waveform (1, num_samples)
extractor = LogMelExtractor(sample_rate=22050, n_mels=128)
spectrogram = extractor(raw_audio_waveform)  # Output: (1, 1, 128, time_steps)

# Apply SpecAugment during training
spec_aug = SpecAugment(freq_mask_param=16, time_mask_param=32)
augmented_spec = spec_aug(spectrogram)
```

### 2. For Saksham (CRNN & Pulse Train Modeling)
You can directly use the dataset loader and trainer engine to train your CRNN:
```python
from src.data.dataset import create_dataloaders, SyntheticInsectDataset
from src.models.crnn import InsectCRNN
from src.utils.trainer import ModelTrainer

# Load dataset (or pass df when real data is ready)
dataset = SyntheticInsectDataset(num_samples=1200, num_classes=40)
train_loader, val_loader, test_loader, _ = create_dataloaders(dataset, batch_size=32)

# Instantiate CRNN model
model = InsectCRNN(num_classes=40, in_channels=1, use_temperature=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Train CRNN
trainer = ModelTrainer(model=model, optimizer=optimizer, model_name="crnn")
trainer.fit(train_loader, val_loader, epochs=10)
```

### 3. For Abhishek (Dataset Curation & Classical MLP Baseline)
- **When your curated dataset is ready**: Provide a CSV with columns `file_path`, `label` (or `species`), and `temperature`. Anyone can load it via:
  ```bash
  python scripts/train_cnn.py --data-csv path/to/metadata.csv
  ```
- **For your Handcrafted MLP Baseline**: Use the feature extractor to generate the 49 classical features (MFCCs, Deltas, Spectral Centroid, Bandwidth, Rolloff, ZCR):
  ```python
  from src.data.features import HandcraftedFeatureExtractor

  extractor = HandcraftedFeatureExtractor(sample_rate=22050)
  features = extractor.extract(raw_audio_numpy)  # Output: (49,) numpy array
  ```

---

## 📁 Folder Structure

```
.
├── configs/
│   ├── vgg_cnn_config.yaml            # VGG CNN hyperparameters
│   └── efficientnetv2_config.yaml     # EfficientNetV2-S config
├── notebooks/
│   ├── 01_eda_and_feature_pipeline.ipynb              # Visualizing Mel spectrograms & SpecAugment
│   └── 02_vgg_cnn_and_efficientnet_training.ipynb     # Training & evaluating CNN models
├── scripts/
│   ├── run_pipeline_demo.py           # Quick sanity check / smoke test
│   ├── train_cnn.py                   # Train VGG-style CNN
│   ├── train_efficientnet.py          # Train EfficientNetV2-S Transfer Learning
│   └── evaluate.py                    # Cross-model evaluation benchmark
├── src/
│   ├── data/
│   │   ├── features.py                # 128-band log-Mel & handcrafted feature extractors
│   │   ├── augmentation.py            # Waveform perturbations & SpecAugment
│   │   ├── dataset.py                 # PyTorch Dataset & DataLoader creator
│   │   └── subset_curator.py          # Subset curation utility
│   ├── models/
│   │   ├── vgg_cnn.py                 # LogMelCNN (VGG-style)
│   │   ├── efficientnet.py            # InsectEfficientNetV2
│   │   ├── crnn.py                    # CRNN template
│   │   ├── mlp_baseline.py            # MLP baseline template
│   │   └── ast_model.py               # AST template
│   └── utils/
│       ├── metrics.py                 # Macro-F1 & Long-Tail Tier analysis
│       └── trainer.py                 # PyTorch Trainer with early stopping & checkpointing
├── requirements.txt                   # Project dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Sanity Check (Self-Test)
```bash
python scripts/run_pipeline_demo.py
```

### 3. Train Models
```bash
# Train VGG-style CNN
python scripts/train_cnn.py --epochs 10 --batch-size 32 --use-temp

# Train EfficientNetV2-S Transfer Learning
python scripts/train_efficientnet.py --epochs 10 --batch-size 16 --use-temp
```

### 4. Evaluate Benchmark Table
```bash
python scripts/evaluate.py
```
