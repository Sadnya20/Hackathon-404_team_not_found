# ✈️ Airline Profit Prediction

![avi](https://github.com/user-attachments/assets/6f3a16ac-3ad3-4491-8553-0ffedbf334bf)

## 🔍 Ever wondered how airlines decide ticket prices, manage costs, and maximize profits?
This project unlocks the secrets behind airline profitability using cutting-edge machine learning and data-driven insights! 🚀

## 🎯 Project Overview
The **Airline Profit Prediction** system predicts airline profits based on real-world financial and operational metrics. It helps airlines make informed decisions, optimize resources, and boost revenue! 🔥

## 📊 Key Highlights

- **🧠 Smart Predictions:** Uses historical airline data to predict future profits.
- **🛠️ Data Processing & Cleaning:** Transforms messy data into meaningful insights.
- **💡 Feature Engineering:** Creates powerful new metrics that improve accuracy.
- **📊 Power BI Dashboards:** Interactive visualizations for deep analysis.
- **🔄 Automated Updates:** Keeps predictions fresh and accurate.

## 🚀 Future Enhancements
We plan to integrate the following:
- **📡 API Deployment on Azure:** Enable real-time profit prediction and easy access.
- **📊 Advanced Power BI Integration:** Enhanced visualization with real-time analytics.
- **🔄 Continuous Model Improvement:** Uploading new datasets to improve model accuracy.

## 🔬 Why Random Forest Regression Over Linear Regression?
We chose **Random Forest Regression** because:
- It captures **non-linear relationships** in airline data.
- It **reduces overfitting** by averaging multiple decision trees.
- It handles **missing and categorical values** better.

Linear Regression assumes a linear relationship, which may not hold in complex airline data.

## ⚖️ Why Standard Scaler Over MinMax Scaler?
We used **Standard Scaler** because:
- It ensures a **zero mean and unit variance**, making it suitable for ML models sensitive to feature scales.
- **MinMax Scaler** can distort data when outliers are present, making it less effective.

---

## 🚀 Quick Start Guide

### Prerequisites
✔️ Python 3.8+  
✔️ Virtual Environment  
✔️ Power BI Desktop  

### Installation
```bash
# Clone the repository
git clone https://github.com/Sadnya20/Airline-Profit-Prediction.git
cd Airline-Profit-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## 🔥 How It Works

### Train the Model
```bash
python main.py --mode train --data_path data/raw/airline_data.csv
```

### Generate Predictions
```bash
python main.py --mode predict --input your_input.json
```

---

## 📊 Power BI Dashboard - **See the Magic!**
Want to explore your airline's profits like a pro? Our **Power BI Dashboard** turns raw data into **beautiful, actionable insights**! 🎨📈

1. **Export Predictions:** Model outputs are stored in `data/processed/predictions.csv`.
2. **Load Data in Power BI:** Import CSV and generate dynamic visual reports.
3. **Automate Updates:** Schedule refreshes for real-time insights.

![](https://your-image-link.com/dashboard.png)

---

![collaborator](https://github.com/user-attachments/assets/28a7a88e-2cc5-4625-b09e-b5cfc498febb)

## 💡 Contributors
- **Sadnya Kolhe** (Team Lead) - [GitHub](https://github.com/Sadnya20/Hackathon-404_team_not_found)
- **Daniyal Sheikh** 
- **Sairaj Ajgaonkar**
- **Contributions welcome! 🎉**

## 💙 What We Learned
This project taught us **collaboration**, **problem-solving**, and the **fun of predictive modeling**! 🚀




