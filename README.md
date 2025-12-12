# Alzheimer's Disease Detection - Deep Learning Project

## Team Information

**Team Name:** NeuroVision

| Name | Neptune Code |
|------|--------------|
| Md Firoz Wadud | IB82MC |
| Hana Becha | KDJ0P0 |
| Wissem ben haj younes | R098MB |

---

## Project Description

This project applies deep learning techniques to classify Alzheimer's disease stages from MRI brain scans. The goal is to develop a model that can accurately categorize brain scans into four classes:

- **Non-Demented**: No signs of Alzheimer's disease
- **Very Mild Demented**: Early stage cognitive decline
- **Mild Demented**: Noticeable symptoms
- **Moderate Demented**: Advanced stage with significant impairment

The project includes complete data preparation, baseline model implementation, custom CNN architecture development, transfer learning experiments, and comprehensive evaluation.

---

## Data Source

### Primary Dataset
- **Source**: [Kaggle - Alzheimer MRI 4-Classes Dataset](https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset)
- **Download Method**: Using Kaggle API with authentication token
- **Format**: MRI brain scan images organized by disease severity class
- **Total Images**: 6,400 images across 4 classes

### How Data Was Downloaded
1. Dataset downloaded from Kaggle using their API
2. Authentication via `kaggle.json` API token
3. Automated extraction and organization into class-specific folders
4. Dataset URL: `https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset`

---

## Repository Structure
```
alzheimer-detection/
│
├── README.md                          # Project documentation
├── documentation.pdf                  # Final project report (6-7 pages)
├── Final_Version.ipynb                # Complete project notebook
├── Milestone_1.ipynb                  # Data preparation workflow
├── Milestone_2.ipynb                  # Baseline models workflow
├── visualization_outputs/             # Data exploration visualizations
├── Alzheimer_MRI_4_classes_dataset/   # Original Dataset From Kaggle
│   ├── MildDemented/
│   ├── ModerateDemented/
│   ├── NonDemented/
│   └── VeryMildDemented/   
│
└── processed_dataset/                 # Preprocessed data
    ├── train/                         # Training set (70%)
    │   ├── MildDemented/
    │   ├── ModerateDemented/
    │   ├── NonDemented/
    │   └── VeryMildDemented/
    ├── val/                           # Validation set (20%)
    │   ├── MildDemented/
    │   ├── ModerateDemented/
    │   ├── NonDemented/
    │   └── VeryMildDemented/
    └── test/                          # Test set (10%)
        ├── MildDemented/
        ├── ModerateDemented/
        ├── NonDemented/
        └── VeryMildDemented/
```

---

## File Descriptions

### `documentation.pdf`
**Final project report (6-7 pages)** containing:
- **Introduction**: Overview of Alzheimer's disease detection, challenges with class imbalance, and project objectives
- **Methods**: Detailed description of NeuroVision V1 (custom CNN) and VGG16 transfer learning architectures, training configuration, and hyperparameter optimization
- **Evaluation**: Comprehensive results on test data with visualizations including training/validation curves, confusion matrix, ROC curves, and baseline comparisons
- **Conclusions**: Main findings showing 97.15% accuracy with custom CNN, effectiveness of class weighting strategy, and practical implications

### `Final_Version.ipynb`
**Complete end-to-end project implementation** including:
- Data acquisition and preprocessing pipeline
- Exploratory data analysis with visualizations
- Baseline model implementation (Logistic Regression and Simple ANN)
- **NeuroVision V1**: Custom CNN architecture with SeparableConv2D layers and progressive dropout
- **VGG16 Transfer Learning**: Pre-trained model with custom classification head
- Training with class weighting, label smoothing, and learning rate scheduling
- Comprehensive evaluation with confusion matrices, ROC curves, and performance metrics
- Weights & Biases integration for experiment tracking
- Final model performance: 97.15% accuracy, 0.9987 AUC

### `Milestone_1.ipynb`
Data preparation notebook containing:
- **Data Acquisition**: Downloads dataset from Kaggle using API
- **Data Exploration**: Visualizes class distributions, sample images, and statistics
- **Data Preprocessing**: Resizes images to 176×176, converts to RGB, normalizes pixel values
- **Dataset Splitting**: Splits data into train (70%), validation (20%), and test (10%) sets
- **Data Export**: Creates downloadable preprocessed dataset

### `Milestone_2.ipynb`
Baseline model development notebook including:

**Baseline Logistic Regression**:
- Feature extraction by flattening 176×176×3 images
- Feature scaling with StandardScaler
- Multinomial Logistic Regression training
- Achieved 92.44% test accuracy

**Baseline Neural Network**:
- Fully connected architecture with 5 hidden layers
- Data augmentation (rotation, shifting, flipping, zoom)
- Early stopping and multiple metrics tracking
- Confusion matrix and ROC curve generation

### `visualization_outputs/`
Contains all generated visualizations:
- Class distribution charts (before/after preprocessing)
- Sample MRI images from each class
- Training and validation loss/AUC curves
- Confusion matrices for all models
- ROC curves comparison
- Heatmaps showing data balance

### `processed_dataset/`
Final preprocessed dataset ready for model training:
- All images resized to 176×176 pixels
- RGB format (3 channels)
- Normalized pixel values (0-1 range)
- Organized by train/val/test splits
- Maintains class balance across splits

---

## How to Run the Solution

### Prerequisites
- Google Colab (recommended) or Jupyter Notebook
- Kaggle account and API token
- Python 3.7+ with TensorFlow 2.x

### Quick Start with Final Version

1. **Open Final_Version.ipynb**
   - Upload to Google Colab or open in Jupyter Notebook
   - This contains the complete project implementation

2. **Get Kaggle API Token**
   - Go to [https://www.kaggle.com](https://www.kaggle.com) and log in
   - Click your profile picture → **Settings** → **API** section
   - Click **"Create New API Token"**
   - `kaggle.json` will download to your computer

3. **Upload API Token**
   - Run the cell that prompts for file upload
   - Upload your `kaggle.json` file

4. **Run All Cells**
   - Execute cells sequentially from top to bottom
   - The notebook will automatically:
     - Download and preprocess the dataset
     - Train baseline models for comparison
     - Train NeuroVision V1 (custom CNN)
     - Train VGG16 transfer learning model
     - Generate all evaluation metrics and visualizations
     - Track experiments with Weights & Biases

5. **View Results**
   - Training curves, confusion matrices, and ROC curves will be displayed
   - Final test accuracy and per-class metrics will be shown
   - Best model weights will be saved

### Step-by-Step Milestones

**For Milestone 1** (Data Preparation):
- Run `Milestone_1.ipynb` for data acquisition and preprocessing only

**For Milestone 2** (Baseline Models):
- Run `Milestone_2.ipynb` for baseline Logistic Regression and ANN models

**For Final Submission** (Complete Project):
- Run `Final_Version.ipynb` for the full implementation with advanced models

---

## Project Results

### Model Performance Summary

| Model | Test Accuracy | Macro F1 | AUC | Parameters |
|-------|--------------|----------|-----|------------|
| **NeuroVision V1 (Custom CNN)** | **97.15%** | 0.9435 | **0.9987** | 1.2M |
| VGG16 Transfer Learning | N/A | N/A | 0.9803 | 15.1M |
| Logistic Regression (Baseline) | 92.44% | 0.90 | N/A | N/A |
| Simple ANN (Baseline) | 52.00% | 0.26 | N/A | N/A |

### NeuroVision V1 Per-Class Performance

| Class | Precision | Recall | F1-Score | AUC | Test Samples |
|-------|-----------|--------|----------|-----|--------------|
| Mild Demented | 0.9524 | 0.9929 | 0.9722 | 0.9999 | 141 |
| Moderate Demented | 0.9000 | 0.8182 | 0.8571 | 0.9929 | 11 |
| Non-Demented | 0.9899 | 0.9609 | 0.9752 | 0.9916 | 511 |
| Very Mild Demented | 0.9562 | 0.9831 | 0.9694 | 0.9899 | 355 |

### Key Achievements
- **97.15% accuracy** on test set with custom CNN architecture
- Successfully handled severe class imbalance (Moderate Demented: only 11 test samples)
- **0.9987 AUC** indicating near-perfect discrimination
- Achieved 0.8571 F1-score on rarest class (Moderate Demented)
- Outperformed all baseline models significantly (+45% over simple ANN, +5% over Logistic Regression)
- Efficient model with only 1.2M parameters

---

## Technical Details

### Data Preprocessing
- **Original Format**: Various sizes, grayscale/RGB
- **Final Format**: 176×176 pixels, RGB (3 channels)
- **Normalization**: Pixel values scaled to [0, 1] range
- **Split Ratio**: 70% train / 20% validation / 10% test
- **Class Weighting**: Applied to handle imbalance (Moderate: 5.0, Mild: 2.0, Very Mild: 1.3, Non: 1.0)

### NeuroVision V1 Architecture
- **Input**: 208×176×3 (adjusted for better aspect ratio preservation)
- **Feature Extraction**: 
  - Initial Conv2D blocks (16 filters)
  - 4 SeparableConv2D blocks (32→64→128→256 filters)
  - Batch Normalization after each block
  - Max pooling for spatial reduction
- **Classification Head**: 
  - Dense layers: 512→128→64 units
  - Progressive dropout: 0.7→0.5→0.3
  - ELU activation throughout
- **Output**: 4-way softmax classification

### Training Configuration
- **Optimizer**: Adam (lr=3×10⁻⁴ for custom CNN, lr=1×10⁻⁴ for transfer learning)
- **Loss**: Categorical cross-entropy with label smoothing (ε=0.15)
- **Regularization**: Gradient clipping (clipnorm=0.5), dropout, batch normalization
- **Callbacks**: ReduceLROnPlateau (factor=0.8, patience=10), EarlyStopping (patience=20)
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Primary Metric**: AUC (robust to class imbalance)

---

## Key Statistics
- **Total Images**: 6,400
- **Training Set**: 4,480 images (70%)
- **Validation Set**: 1,280 images (20%)
- **Test Set**: 1,018 images (10%)
- **Image Dimensions**: 208 × 176 × 3 (final), 176 × 176 × 3 (milestones)
- **Classes**: NonDemented (50.2%), VeryMildDemented (34.9%), MildDemented (13.8%), ModerateDemented (1.1%)

---

## References

- **Dataset**: [Alzheimer MRI 4-Classes Dataset - Kaggle](https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset)
- **Baseline Reference**: [DL-Simplified Alzheimer's Detection](https://github.com/abhisheks008/DL-Simplified/tree/main/Alzheimers%20Detection)
- **Key Papers**:
  - Litjens et al., "A survey on deep learning in medical image analysis," Medical Image Analysis, 2017
  - Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," CVPR, 2017
  - Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR, 2015

---

## Course Information

- **Course**: Deep Learning  
- **Instructor**: Al-Radhi Mohammed Salah Hamza  
- **Institution**: Budapest University of Technology and Economics (BME)    
- **Submission Date**: December 2025    

---

## Acknowledgments

We would like to thank:
- Marco Pinamonti for providing the Alzheimer's MRI dataset on Kaggle
- Our course instructor for guidance and support throughout the project
- The deep learning community for open-source tools and resources

---
