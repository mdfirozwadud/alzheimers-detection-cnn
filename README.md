# Alzheimer's Disease Detection using Deep Learning

**Project Milestone 1: Data Acquisition and Preparation**

---

## Team Information

**Team Name:** 

| Name | Neptune Code |
|------|--------------|
| Md Firoz Wadud | IB82MC |
| Hana Becha | KDJ0P0 |
| Wissem ben haj younes | R098MB |

---

## Project Overview

Alzheimer's disease (AD) is a progressive neurodegenerative disorder that significantly impacts memory and cognitive abilities. Early detection is crucial for improving patient care and enabling timely medical intervention. This project explores the application of deep learning models for classifying Alzheimer's disease stages from MRI brain scans.

### Objectives

This milestone focuses on the foundational steps of our deep learning pipeline:

1. **Data Acquisition**: Sourcing and downloading MRI datasets from reliable repositories
2. **Data Exploration**: Analyzing dataset characteristics, distributions, and patterns
3. **Data Preparation**: Preprocessing images and organizing data for model training
4. **Dataset Splitting**: Creating properly balanced training, validation, and test sets

### Classification Task

Our model aims to classify MRI scans into four distinct categories:
- **Non-Demented**: No signs of Alzheimer's disease
- **Mild Demented**: Early stage of cognitive decline
- **Moderate Demented**: Intermediate stage with noticeable symptoms
- **Severe Demented**: Advanced stage with significant impairment

### Dataset Sources

We utilize a Kaggle datasets for this project:

1. **Primary Dataset**: [Alzheimer MRI 4-Classes Dataset](https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset)

**Baseline Framework**: We build upon the foundation provided by the [DL-Simplified Alzheimer's Detection Repository](https://github.com/abhisheks008/DL-Simplified/tree/main/Alzheimers%20Detection)

---

## Repository Structure

```
alzheimer-detection-milestone1/
│
├── README.md                          # Project documentation
│
├── Milesstone_1.ipynb
│
└── data/
    ├── raw/                          # Original downloaded datasets
    ├── processed/                    # Preprocessed images
    └── splits/                       # Train/validation/test splits

```

---

## File Descriptions

### Notebooks

**`milestone1_data_preparation.ipynb`**
- Complete workflow for Milestone 1
- Includes data acquisition, exploration, preprocessing, and splitting
- Contains visualizations and statistical analysis
- Outputs training, validation, and test datasets

### Source Code Modules


### Data Directory

-  **`data/raw/`**: Original MRI images organized by class
- **`data/processed/`**: Preprocessed images ready for model training
- **`data/splits/`**: Final train/validation/test datasets with associated labels

---

## How to Run the Solution
1. Open the Milesstone_1.ipynb file in google colab or jupyter notebook
2. Run the  Imports
3. Run the Download dataset Section. Here the dataset will be downloaded from kaggle but you need to upoload a api tocke.
4. how to get the api tocken 
 a.Go to https://www.kaggle.com and log in
b.Click your profile picture (top right) → Settings
c.Scroll to API section
d.Click "Create New API Token"
e.kaggle.json downloads automatically to your Downloads folder
4. Then upload the kaggle.json file
5. run the next blocks



---

## Milestone 1 Deliverables

Upon successful execution, this notebook produces:

1. **Downloaded Datasets**: Raw MRI images from Kaggle sources
2. **Exploratory Analysis**: 
   - Class distribution charts
   - Sample images from each category
   - Statistical summaries
3. **Preprocessed Data**: 
   - Resized and normalized images
   - Consistent format across all samples
4. **Dataset Splits**:
   - Training set (70% of data)
   - Validation set (20% of data)
   - Test set (10% of data)
   - All splits maintain class balance

### Output Files
Preprocessed dataset
notebook file
readme

---

## References

- Baseline Repository: [DL-Simplified Alzheimer's Detection](https://github.com/abhisheks008/DL-Simplified/tree/main/Alzheimers%20Detection)
- Dataset 1: [Kaggle - Alzheimer MRI 4-Classes](https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset)
- Dataset 2: [Kaggle - Alzheimer's Dataset 4 Classes](https://www.kaggle.com/datasets/preetpalsingh25/alzheimers-dataset-4-class-of-images)

---

**Course**: Deep Learning  
**Institution**: Budapest Institute of Technology and Economics 
**Submission Date**: October 20, 2025  
