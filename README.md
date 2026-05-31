# 🌾 Crop Recommendation System

## 📌 Project Overview
An end-to-end Machine Learning project that recommends 
the best crop based on soil and climate conditions.

## 🎯 Problem Statement
Farmers struggle to choose the right crop based on 
soil nutrients and climate conditions. This ML model 
solves that problem.

## 🛠️ Tech Stack
- Python
- Jupyter Notebook
- KNN Classifier (scikit-learn)
- MinMaxScaler
- joblib
- Streamlit
- VS Code

## 📊 Dataset
- 2200 rows
- 7 input features (N, P, K, Temperature, Humidity, pH, Rainfall)
- 22 crop types
- Source: Crop_recommendation.csv

## 📁 Project Files
| File | Description |
|---|---|
| app.py | Streamlit web application |
| crop_recommendation.ipynb | Jupyter Notebook (ML pipeline) |
| crop_model_production.pkl | Trained model + scaler |
| Crop_recommendation.csv | Dataset |
| requirements.txt | Required libraries |

## ⚙️ How to Run
1. Clone this repository
2. Create virtual environment: python -m venv venv
3. Activate: venv\Scripts\activate
4. Install libraries: pip install -r requirements.txt
5. Run app: streamlit run app.py

## 📈 Model Performance
- Algorithm: KNN Classifier
- Accuracy: 99.09%

## 🌱 Input Features
- N — Nitrogen content (kg/ha)
- P — Phosphorus content (kg/ha)
- K — Potassium content (kg/ha)
- Temperature (°C)
- Humidity (%)
- Soil pH
- Rainfall (mm)
