# Alzheimer's Disease Detection - Milestone 1: Data Preparation

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

**Milestone 1** focuses on data acquisition, exploration, preprocessing, and preparation for model training.

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
alzheimer-detection-milestone1/
│
├── README.md                          # Project documentation
├── Milestone_1.ipynb                  # Main notebook with complete workflow
├── Milestone_2.ipynb                  # Main notebook with complete workflow
├── visualization_outputs/             # Data exploration visualizations
├──Alzheimer_MRI_4_classes_dataset/    # Original Dataset From Kaggle
│   ├── MildDemented/
│   ├── ModerateDemented/
│   ├── NonDemented/
│   └── VeryMildDemented/   
│
└── processed_dataset/                 # Preprocessed data (download link below)
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

### `Milestone_1.ipynb`
Main Jupyter Notebook containing the complete Milestone 1 workflow:
- **Data Acquisition**: Downloads dataset from Kaggle using API
- **Data Exploration**: Visualizes class distributions, sample images, and statistics
- **Data Preprocessing**: Resizes images to 176×176, converts to RGB, normalizes pixel values
- **Dataset Splitting**: Splits data into train (70%), validation (20%), and test (10%) sets
- **Data Export**: Creates downloadable preprocessed dataset

**`Milestone_2.ipynb`**
Milestone_1 + Baseline and deep learning model training and evaluation:

Baseline model implementation using Logistic Regression:
- **Data Loading**: Loads preprocessed train/val/test datasets using ImageDataGenerator
- **Feature Extraction**: Flattens 176×176×3 images into 92,928-dimensional feature vectors
- **Feature Scaling**: Standardizes features using StandardScaler for improved convergence
- **Model Training**: Trains multinomial Logistic Regression classifier with LBFGS solver
- **Evaluation**: Generates confusion matrix, classification report, and ROC curves
- **Model Export**: Saves trained baseline model and scaler as pickle files

Deep neural network implementation for Alzheimer's classification:
- **Data Augmentation**: Applies rotation, shifting, flipping, and zoom transformations to training data
- **Model Architecture**: Implements fully connected neural network with 5 hidden layers (100-200-200-200-200 neurons)
- **Training Configuration**: Uses Adam optimizer, categorical cross-entropy loss, early stopping (patience=10)
- **Multi-Metric Tracking**: Monitors accuracy, AUC, precision, and recall during training
- **Evaluation**: Generates confusion matrix, classification report, and ROC curves
- **Training History**: Visualizes loss and accuracy curves across epochs


### `visualization_outputs/`
Contains all generated visualizations:
- Class distribution charts (before/after preprocessing)
- Sample MRI images from each class
- Heatmaps showing data balance
- Comparison plots

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

### Step-by-Step Instructions

1. **Open the Notebook**
```
   File: Milestone_1.ipynb
   Platform: Google Colab or Jupyter Notebook
```

2. **Get Kaggle API Token**
   - Go to [https://www.kaggle.com](https://www.kaggle.com) and log in
   - Click your profile picture (top right) → **Settings**
   - Scroll to **API** section
   - Click **"Create New API Token"**
   - `kaggle.json` will download to your computer

3. **Upload API Token**
   - In the notebook, run the cell that prompts for file upload
   - Upload your `kaggle.json` file

4. **Run All Cells**
   - Execute cells sequentially from top to bottom
   - The notebook will:
     - Download the dataset from Kaggle
     - Perform data exploration and visualization
     - Preprocess all images
     - Split into train/val/test sets
     - Generate downloadable preprocessed dataset

5. **Download Preprocessed Dataset**
   - After completion, the notebook will automatically download `preprocessed_dataset.zip`
   - This contains all training, validation, and test data ready for modeling


---

## Milestone 1 Outputs

### 1. Training Data
- **Location**: `processed_dataset/train/`
- **Size**: 70% of total dataset (~4,480 images)
- **Format**: 176×176 RGB images, normalized (0-1)
- **Classes**: 4 disease severity categories

### 2. Validation Data
- **Location**: `processed_dataset/val/`
- **Size**: 20% of total dataset (~1,280 images)
- **Format**: Same as training data
- **Purpose**: Model hyperparameter tuning

### 3. Test Data
- **Location**: `processed_dataset/test/`
- **Size**: 10% of total dataset (~640 images)
- **Format**: Same as training data
- **Purpose**: Final model evaluation

### 4. Visualizations
- Class distribution plots (before/after preprocessing)
- Sample images from each disease category
- Statistical summaries and heatmaps
- All saved in `visualization_outputs/` folder

---

## Data Exploration Summary

### Preprocessing Details
- **Original Format**: Various sizes, grayscale/RGB
- **Final Format**: 176×176 pixels, RGB (3 channels)
- **Normalization**: Pixel values scaled to [0, 1] range
- **Split Ratio**: 70% train / 20% validation / 10% test
- **Random Seed**: 42 (for reproducibility)

### Key Statistics
- **Total Images**: 6,400
- **Training Set**: ~4,480 images
- **Validation Set**: ~1,280 images
- **Test Set**: ~640 images
- **Image Dimensions**: 176 × 176 × 3
- **Classes**: NonDemented, VeryMildDemented, MildDemented, ModerateDemented


---

## References

- **Dataset**: [Alzheimer MRI 4-Classes Dataset - Kaggle](https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset)
- **Baseline**: [DL-Simplified Alzheimer's Detection](https://github.com/abhisheks008/DL-Simplified/tree/main/Alzheimers%20Detection)

---

## Course Information

-**Course**: Deep Learning  
-**Instructor**: Al-Radhi Mohammed Salah Hamza  
-**Institution**: Budapest University of Technology and Economics (BME)    
-**Submission Date**: October 20, 2025    

---
