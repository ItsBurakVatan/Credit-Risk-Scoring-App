# Credit Scoring Application for Bank Risk Analysis

This project is a **credit scoring and customer risk analysis application** developed specifically for the **Assistant Inspector Interview at Ziraat Bank**. It uses a PostgreSQL database, machine learning with logistic regression, and an interactive Streamlit dashboard to analyze customer risk and credit scoring.

## 🔍 Key Features

- **Interactive Streamlit Dashboard**: For exploring data, training models, and predicting customer risk.
- **Database Integration**: Reads customer data directly from a PostgreSQL database.
- **Machine Learning**: Trains a logistic regression model to predict credit default.
- **Explainability**: SHAP values used to visualize model explanations.
- **Scoring System**: Converts probability into a custom score (0–160) and interprets risk levels.

## 📁 Project Structure

```

├── app.py                    # Main Streamlit application
├── config/
│   └── config.py             # Configuration (DB settings, paths)
├── data/
│   └── db\_connector.py       # Functions for PostgreSQL connection
├── eda/
│   └── eda\_utils.py          # Visualization utilities
├── model/
│   ├── trainer.py            # Model training logic
│   ├── predictor.py          # Prediction and ROC/PR visualization
│   └── explainer.py          # SHAP explainability
├── utils/
│   └── preprocess.py         # Data preprocessing (scaling)
├── model/
│   └── saved\_model.pkl       # Trained ML model
├── requirements.txt          # Python dependencies

````

## 💡 Usage

1. Make sure you have PostgreSQL installed and running.
2. Update the `config/config.py` file with your DB credentials.
3. Run the app:

```bash
streamlit run app.py
````

## 📊 Risk Score Interpretation

| Score Range | Risk Level        |
| ----------- | ----------------- |
| 152–160     | 🔵 Excellent      |
| 144–151     | 🟢 Very Good      |
| 112–143     | 🟡 Medium         |
| 80–111      | 🟠 Low            |
| 0–79        | 🔴 Very High Risk |

## 🏦 Project Motivation

The project was designed to simulate a real-world banking scenario where a financial institution needs to evaluate the likelihood of a customer defaulting on a credit payment. It demonstrates:

* Database connectivity and management
* Applied machine learning for classification
* Risk interpretation through both score and probability
* Visual and explainable results for decision makers

---

**Developed for Ziraat Bank Assistant Inspector Interview.**
