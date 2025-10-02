# GlobalMart_Churn_Project

#  GlobalMart Customer Churn Prediction

This project builds and compares machine learning models to predict customer churn for **GlobalMart**. It uses advanced algorithms (Random Forest, XGBoost) and compares them against a baseline Logistic Regression model.

A Streamlit app is also included for interactive churn prediction and model insights.

---

##  Features

* **Data Preprocessing:** Handling categorical and numerical features, scaling, and encoding.
* **Model Training:** Logistic Regression, Random Forest, and XGBoost.
* **Advanced Evaluation:** Precision, Recall, F1-score, Accuracy, and ROC-AUC.
* **Comparison Report:** Side-by-side evaluation of models with recommendations.
* **Streamlit App:** Interactive UI for predictions and model exploration.

---

##  Project Structure

```
GlobalMart_Churn_Project/
│── data/                  # Dataset (place Telco Customer Churn CSV here)
│── models/                # Saved models
│── notebooks/             # Jupyter notebooks for EDA and training
│── src/                   # Source code
│   ├── preprocessing.py   # Data preprocessing pipeline
│   ├── train.py           # Training utilities
│   └── evaluate.py        # Evaluation functions
│── app.py                 # Streamlit app
│── run_compare.py         # Script to compare models
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

---

##  Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/GlobalMart_Churn_Project.git
   cd GlobalMart_Churn_Project
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the dataset (`Telco-Customer-Churn.csv`) inside the `data/` folder.

---

##  Running the Project

### 1. Compare Models (Logistic Regression vs Random Forest vs XGBoost)

```bash
python run_compare.py --data data/Telco-Customer-Churn.csv --target Churn --test-size 0.25 --preferred-metric f1
```

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```

---

##  Expected Output

* A classification report with Precision, Recall, F1-score, Accuracy, and ROC-AUC.
* A recommendation on which model is best for deployment.
* An interactive Streamlit dashboard where you can:

  * Upload customer data.
  * Get churn predictions.
  * View feature importance and model performance.

---

##  Dataset

The dataset used is the **Telco Customer Churn Dataset** (available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).

---

##  Future Improvements

* Add hyperparameter tuning with GridSearchCV.
* Save best model with joblib/pickle and load into Streamlit app.
* Deploy app to **Streamlit Cloud / Heroku / Docker**.

---

## Author

Muhammad Asad
BS Artificial Intelligence – University of Agriculture Peshawar
Email: [muhammadasad9941@gmail.com](mailto:muhammadasad9941@gmail.com)
