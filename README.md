# road-safety-image-classifier-BD

## Overview

This project uses transfer learning with fastai and PyTorch to automatically classify road safety hazards from images captured in Bangladesh. The classifier can identify 9 distinct road safety conditions, helping raise awareness of traffic and infrastructure risks.
To the best of our knowledge, this is one of the first open-source image classifiers on GitHub that combines multiple Bangladesh-specific road safety hazard categories — including flooded roads, illegal parking, overloaded rickshaws, crowded footpaths, and more — into a single multi-class deep learning model with a live interactive demo.
While related work exists in Bangladesh (e.g. road damage detection, traffic sign datasets, and individual vehicle recognition), no publicly available GitHub project has tackled this broader combination of urban road safety hazards under one classifier. This project aims to fill that gap.

---

## Classes (9 Categories)

| # | Category |
|---|----------|
| 1 | Bangladesh Cracked or Damaged Asphalt Road |
| 2 | Bangladesh Crowded Pedestrian Footpath |
| 3 | Bangladesh Flooded Road |
| 4 | Bangladesh Pedestrians Crossing Busy Roads |
| 5 | Bangladesh Roads Potholes |
| 6 | Bus Stopping in Road |
| 7 | CNG-Autorickshaw in Traffic |
| 8 | Illegal Parking on Roadside in Bangladesh |
| 9 | Overloaded Rickshaw or Van |

---

## Dataset

- **Total images collected:** ~1,051
- **Train / Validation split:** 80% / 20% (random split, seed=42)
- **Data collection:** Bing Image Crawler (keyword-based per category)
- **Data cleaning:** Misclassified and corrupted images were removed using fastai's `ImageClassifierCleaner` before training all models

## Sample Data
<p align="center">

  <img src="test_images/image_2.jpg" width="300"/>

</p>
Figure: Bangladesh Flooded Road

## Models Trained & Results

Three architectures were evaluated using fastai's `vision_learner` with fine-tuning. All models were trained on the same cleaned dataset. Metrics reported are **weighted averages** on the validation set at the final epoch.

### Model 1 — ResNet-18 (20 epochs)

| Metric | Score |
|--------|-------|
| Accuracy | **89.52%** |
| Error Rate | 10.48% |
| Precision | 89.69% |
| Recall | 89.52% |
| F1 Score | 89.38% |

### Model 2 — ResNet-34 (20 epochs)

| Metric | Score |
|--------|-------|
| Accuracy | **89.52%** |
| Error Rate | 10.48% |
| Precision | 89.98% |
| Recall | 89.52% |
| F1 Score | 89.13% |

### Model 3 — DenseNet-121 (20 epochs) Best Model

| Metric | Score |
|--------|-------|
| Accuracy | **90.95%** |
| Error Rate | 9.05% |
| Precision | 91.48% |
| Recall | 90.95% |
| F1 Score | 90.91% |

---

## Model Comparison Summary

| Model | Epochs | Accuracy | F1 Score |
|-------|--------|----------|----------|
| ResNet-18 | 20 | 89.52% | 89.38% |
| ResNet-34 | 20 | 89.52% | 89.13% |
| **DenseNet-121** | **20** | **90.95%** | **90.91%** |

### Why DenseNet-121 Performed Best

**DenseNet-121** achieved the highest accuracy (90.95%) and F1 score (90.91%) among all three models. DenseNet's architecture connects each layer to every other layer in a feed-forward fashion — this dense connectivity allows the network to reuse features at multiple levels, which is particularly effective when the dataset is relatively small (~1,000 images). Compared to ResNet-18 and ResNet-34, DenseNet-121 made better use of the limited training data by propagating gradients more efficiently and reducing the risk of vanishing gradients. The result is a more expressive model that generalises better without needing significantly more parameters.

---

## Project Structure

```
road-safety-image-classifier-BD/
├── Data_Loader/          # Data loading and augmentation notebook
├── Dataset_Classifier/   # Training notebooks (ResNet-18, ResNet-34, DenseNet-121)
├── Saved_model/          # Exported .pkl model files
├── classifier_app/       # Gradio inference app
├── docs/                 # GitHub Pages website
├── test_images/          # Sample images for testing
├── README.md
└── LICENSE
```

---

## How to Run the Classifier App

```bash
pip install fastai gradio
python classifier_app/Road_Safety_Classifier.py
```

Or open the hosted live demo on GitHub Pages.

---

## Tech Stack

- **Python** 3.12
- **fastai** 2.8.7
- **PyTorch** 2.10.0
- **Gradio** (for the interactive demo)
- **DenseNet-121** (best performing model, pre-trained on ImageNet)

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
