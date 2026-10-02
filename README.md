# Insect Acoustic Classification (ICT 4442 Mini Project)

**Branch**: `saurabh`  
**Author**: Saurabh Tiwari (230911238)

Hey team, I've built the shared audio preprocessing pipeline, spectrogram extraction, augmentations, and the CNN baselines (VGG-style CNN and EfficientNetV2-S). 

Everything is modular so you can directly import the data loaders, feature extractors, and trainer for your own models (AST, CRNN, and MLP).

---

## What is in this branch

1. **Audio Feature Extraction (`src/data/features.py`)**:
   - `LogMelExtractor`: Converts 5-second audio chunks into 128-band log-Mel spectrograms.
   - `HandcraftedFeatureExtractor`: Extracts 49 features (MFCCs + deltas, spectral centroid, bandwidth, rolloff, zero-crossing rate) for the classical MLP baseline.

2. **Data Augmentations (`src/data/augmentation.py`)**:
   - Audio waveform: Time shifting, random noise, gain adjustments.
   - Spectrogram: SpecAugment (frequency and time masking).

3. **Dataset & Loader (`src/data/dataset.py`)**:
   - `InsectAudioDataset`: Loads real audio files from a metadata CSV and chunks them into 5s spectrograms.
   - `SyntheticInsectDataset`: Generates synthetic insect calls (calibrated with temperature shift) so we can test and train models locally without waiting for the full 200GB download.
   - `create_dataloaders`: Creates train/val/test splits (60/20/20).

4. **CNN Models (`src/models/`)**:
   - `LogMelCNN` in `vgg_cnn.py`: 4-block VGG-style CNN with dropout, batchnorm, and temperature conditioning.
   - `InsectEfficientNetV2` in `efficientnet.py`: Pretrained EfficientNetV2-S transfer learning model.

5. **Trainer & Evaluation (`src/utils/`)**:
   - `ModelTrainer`: Training loop with learning rate scheduling and early stopping.
   - `calculate_metrics`: Computes Macro-F1, Accuracy, and long-tail performance breakdown (Head / Medium / Tail classes).

---

## How to use this in your models

### 1. Harsh (Audio Spectrogram Transformer - AST)
To extract 128-band log-Mel spectrograms and apply SpecAugment for your transformer:
```python
import torch
from src.data.features import LogMelExtractor
from src.data.augmentation import SpecAugment

# Audio input: (batch_size, num_samples) at 22050 Hz
extractor = LogMelExtractor(sample_rate=22050, n_mels=128)
spec = extractor(raw_audio)  # returns tensor of shape (batch, 1, 128, time_steps)

# SpecAugment for training
spec_aug = SpecAugment(freq_mask_param=16, time_mask_param=32)
augmented_spec = spec_aug(spec)
```

### 2. Saksham (CRNN & Sequence Modeling)
You can reuse the dataset loader and training loop directly:
```python
import torch
from src.data.dataset import create_dataloaders, SyntheticInsectDataset
from src.models.crnn import InsectCRNN
from src.utils.trainer import ModelTrainer

dataset = SyntheticInsectDataset(num_samples=1200, num_classes=40)
train_loader, val_loader, test_loader, _ = create_dataloaders(dataset, batch_size=32)

model = InsectCRNN(num_classes=40, in_channels=1, use_temperature=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

trainer = ModelTrainer(model=model, optimizer=optimizer, model_name="crnn")
trainer.fit(train_loader, val_loader, epochs=10)
```

### 3. Abhishek (Dataset Curation & MLP Baseline)
- When you finish curating the InsectSet459 subset, just save a CSV with `file_path`, `label`, and `temperature`. Anyone can pass it via `--data-csv`:
  ```bash
  python scripts/train_cnn.py --data-csv path/to/metadata.csv
  ```
- For your handcrafted MLP baseline, use the feature extractor:
  ```python
  from src.data.features import HandcraftedFeatureExtractor

  extractor = HandcraftedFeatureExtractor(sample_rate=22050)
  features = extractor.extract(audio_array)  # returns 49-dim numpy vector
  ```

---

## Quick Start

### 1. Install packages
```bash
pip install -r requirements.txt
```

### 2. Run sanity check
To test if all modules, feature extraction, and models are working properly:
```bash
python scripts/run_pipeline_demo.py
```

### 3. Training commands
```bash
# Train VGG-style CNN
python scripts/train_cnn.py --epochs 10 --batch-size 32 --use-temp

# Train EfficientNetV2-S
python scripts/train_efficientnet.py --epochs 10 --batch-size 16 --use-temp
```

### 4. Evaluate all models together
```bash
python scripts/evaluate.py
```
