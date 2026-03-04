# Heart Disease Prediction Using Secure Federated Learning

## Overview

This project implements a privacy-preserving Heart Disease Prediction system using **Federated Learning (FL)**. Instead of centralizing sensitive medical data, multiple institutions train models locally and share only encrypted model updates for aggregation.

The objective is to demonstrate how machine learning can be deployed in healthcare environments while preserving patient privacy and maintaining strong predictive performance.

## Objectives

- Build a binary classification model to predict heart disease
- Implement federated learning for decentralized model training
- Ensure data privacy through secure aggregation
- Evaluate model performance without sharing raw patient data

## Key Features

- Federated learning architecture
- Secure model parameter aggregation
- Decentralized training simulation
- Binary classification (Heart Disease: Yes / No)
- Real-time monitoring support

## System Architecture

1. **Local Training (Hospital Nodes)**
   - Each node trains a local model using its private dataset.
   - No raw patient data leaves the institution.

2. **Encrypted Parameter Sharing**
   - Only model weights are transmitted to the central server.

3. **Federated Averaging**
   - Central server aggregates weights using weighted averaging.

4. **Global Model Redistribution**
   - Updated global model is sent back to each node for further training.

## Dataset Features

The dataset includes structured clinical attributes such as:

- Age
- Sex
- Resting Blood Pressure
- Cholesterol
- Maximum Heart Rate
- Chest Pain Type
- Fasting Blood Sugar
- ECG Results
- Exercise-Induced Angina
- ST Depression

**Target Variable:** Presence (1) or Absence (0) of Heart Disease

## Tech Stack

- Python
- Scikit-learn
- TensorFlow / PyTorch
- Pandas
- NumPy
- Custom Federated Learning Simulation

## Methodology

### 1️⃣ Data Preprocessing
- Handling missing values
- Feature scaling (StandardScaler)
- Categorical encoding

### 2️⃣ Local Model Training
- Logistic Regression / Neural Network classifier
- Local training on decentralized datasets

### 3️⃣ Secure Aggregation
- Weighted averaging of model parameters
- Iterative global model updates

### 4️⃣ Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

## Privacy & Security

- No raw data sharing
- Encrypted weight transmission
- Secure model aggregation protocol
- Designed for healthcare compliance environments

## Future Improvements

- Differential Privacy integration
- Homomorphic Encryption support
- Model explainability (SHAP / LIME)
- Deployment-ready federated server
- Multi-institution real-world validation

## Key Takeaway

This project demonstrates that federated learning enables collaborative healthcare AI without compromising patient privacy. It balances predictive performance with regulatory-conscious system design, making it suitable for real-world medical AI deployment.
