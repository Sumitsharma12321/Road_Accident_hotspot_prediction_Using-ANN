# 🚦 Road Accident Hotspot Prediction Using ANN

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Dataset-2M%2B%20Records-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Accuracy-89.84%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  A deep learning system that predicts road accident hotspots using an Artificial Neural Network (ANN) trained on over 2 million real accident records. Grid-based geo-encoding is used to label dangerous zones, enabling authorities to take targeted preventive action.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Future Scope](#-future-scope)
- [License](#-license)
- [Author](#-author)

---

## 📖 Overview

Road accidents are a critical public safety issue. This project uses an **Artificial Neural Network (ANN)** to classify geographic grid cells as **accident hotspots or non-hotspots** based on historical accident data. A grid-based geo-encoding approach converts raw latitude/longitude coordinates into spatial grid cells, and a hotspot label is assigned based on accident frequency within each cell.

---

## 🎯 Problem Statement

Given historical road accident data containing location, weather, road type, lighting, and time information — can we train an ANN to accurately identify which geographic zones are **accident hotspots** (areas with a high density of accidents)?

---

## 📂 Dataset

- **File:** `Accident_Information.csv`
- **Total Records:** 2,047,256 rows × 34 columns
- **Records after cleaning:** 2,046,925

### Selected Features

| Column                | Description                               |
|-----------------------|-------------------------------------------|
| `Latitude`            | Latitude of the accident location         |
| `Longitude`           | Longitude of the accident location        |
| `Accident_Severity`   | Severity level (Slight / Serious / Fatal) |
| `Weather_Conditions`  | Weather at the time of accident           |
| `Road_Type`           | Type of road (single carriageway, etc.)   |
| `Light_Conditions`    | Lighting (daylight, dark, etc.)           |
| `Time`                | Time of accident (extracted as hour 0–23) |
| `Urban_or_Rural_Area` | Urban or rural classification             |

---

## 🔄 Project Workflow

```
Raw CSV Data (2,047,256 records)
        │
        ▼
  Select 8 Key Features
        │
        ▼
  Drop Null Values
        │
        ▼
  Grid-Based Geo Encoding
  (grid_size = 0.02° ≈ 2 km cells)
        │
        ▼
  Create Hotspot Label
  (is_hotspot = 1 if acc_count > 30)
        │
        ▼
  Label Encode Categorical Columns
  + Extract Hour from Time column
        │
        ▼
  Train / Test Split  (80% / 20%)
        │
        ▼
  StandardScaler Normalization
        │
        ▼
  ANN Model Training (10 Epochs, batch=256)
        │
        ▼
  Evaluation + Hotspot Scatter Map
```

---

## 🧠 Model Architecture

The ANN is built using the **TensorFlow / Keras Sequential API**:

```
Input Layer      →  8 features
Hidden Layer 1   →  Dense(128, activation='relu')
                    Dropout(0.3)
Hidden Layer 2   →  Dense(64,  activation='relu')
                    Dropout(0.3)
Hidden Layer 3   →  Dense(32,  activation='relu')
Output Layer     →  Dense(1,   activation='sigmoid')
```

**Compilation Settings:**

| Parameter      | Value                |
|----------------|----------------------|
| Optimizer      | Adam                 |
| Loss Function  | Binary Cross-Entropy |
| Metric         | Accuracy             |
| Epochs         | 10                   |
| Batch Size     | 256                  |
| Validation Set | 20% of data          |

---

## 📊 Results

### Training History

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|-------|----------------|--------------|------------|----------|
| 1     | 88.38%         | 89.27%       | 0.2397     | 0.2284   |
| 2     | 89.05%         | 89.39%       | 0.2295     | 0.2239   |
| 3     | 89.21%         | 89.55%       | 0.2265     | 0.2212   |
| 4     | 89.29%         | 89.55%       | 0.2247     | 0.2194   |
| 5     | 89.37%         | 89.69%       | 0.2236     | 0.2185   |
| 6     | 89.41%         | 89.74%       | 0.2227     | 0.2172   |
| 7     | 89.47%         | 89.86%       | 0.2221     | 0.2168   |
| 8     | 89.49%         | 89.81%       | 0.2215     | 0.2169   |
| 9     | 89.56%         | 89.86%       | 0.2209     | 0.2159   |
| 10    | 89.58%         | **89.84%**   | 0.2206     | **0.2155** |

### Final Test Evaluation

| Metric       | Value      |
|--------------|------------|
| **Accuracy** | **89.84%** |
| Loss         | 0.2155     |

### Classification Report

| Class              | Precision | Recall | F1-Score | Support   |
|--------------------|-----------|--------|----------|-----------|
| 0 (Non-Hotspot)    | 0.65      | 0.67   | 0.66     | 60,477    |
| 1 (Hotspot)        | 0.94      | 0.94   | 0.94     | 3,48,908  |
| **Accuracy**       |           |        | **0.90** | 4,09,385  |
| Macro Avg          | 0.80      | 0.80   | 0.80     | 4,09,385  |
| Weighted Avg       | 0.90      | 0.90   | 0.90     | 4,09,385  |

### Confusion Matrix

```
                        Predicted: Non-Hotspot    Predicted: Hotspot
Actual: Non-Hotspot          40,470                  20,007
Actual: Hotspot              21,580                 3,27,328
```

---

## 📉 Visualizations

The notebook generates the following plots:

**1. 🗺️ Hotspot Scatter Map**
- Plots all accident records by Longitude (X) vs Latitude (Y)
- Color-coded using `coolwarm` palette:
  - 🔵 Blue → Non-Hotspot (`is_hotspot = 0`)
  - 🔴 Red  → Hotspot (`is_hotspot = 1`)

**2. 📈 Training Curves (Accuracy & Loss)**
- Left plot: Train vs Validation Accuracy across 10 epochs
- Right plot: Train vs Validation Loss across 10 epochs
- Both show consistent convergence with no overfitting

---

## 🛠 Tech Stack

| Library           | Purpose                              |
|-------------------|--------------------------------------|
| `pandas`          | Data loading and manipulation        |
| `numpy`           | Numerical operations                 |
| `matplotlib`      | Training curve plots                 |
| `seaborn`         | Hotspot scatter visualization        |
| `scikit-learn`    | Label encoding, scaling, metrics     |
| `tensorflow/keras`| ANN model building and training      |

---

## 📁 Project Structure

```
Road_Accident_hotspot_prediction_Using-ANN/
│
├── Accident_Information.csv        # Dataset (2M+ records)
├── road_accident_hotspot.ipynb     # Main Jupyter Notebook
└── README.md                       # Project documentation
```

---

## ⚙️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Sumitsharma12321/Road_Accident_hotspot_prediction_Using-ANN.git
cd Road_Accident_hotspot_prediction_Using-ANN
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### 3. Add the Dataset

Place `Accident_Information.csv` in the project root, or update the file path inside the notebook:

```python
df = pd.read_csv("/your/path/to/Accident_Information.csv", engine='python', on_bad_lines='skip')
```

### 4. Run the Notebook

```bash
jupyter notebook road_accident_hotspot.ipynb
```

> 💡 **Recommended:** Run on **Google Colab** with GPU enabled for faster training on the 2M+ record dataset.

---

## 🔭 Future Scope

- 🌐 **Web App Deployment** — Wrap the model in a Flask or Streamlit app for real-time predictions.
- 🗺️ **Interactive Heatmaps** — Use Folium or Plotly to render dynamic hotspot heatmaps on real maps.
- 📡 **Live Data Integration** — Feed real-time weather and traffic API data for dynamic hotspot updates.
- ⚖️ **Class Imbalance Handling** — Apply SMOTE or class weights to improve Non-Hotspot (Class 0) recall.
- 🔔 **Authority Alerts** — Automatically notify traffic police when a zone is classified as a hotspot.
- 📈 **Advanced Models** — Explore LSTM or Graph Neural Networks for richer spatiotemporal patterns.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Sumit Sharma**

- 🔗 GitHub: [@Sumitsharma12321](https://github.com/Sumitsharma12321)

---

<p align="center">⭐ If you found this project helpful, please give it a star!</p>
