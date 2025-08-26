# Credit Risk Scoring App

Credit scoring and customer risk analysis application developed for financial risk assessment.  
Built with Streamlit, PostgreSQL, and machine learning.

## Key Features

- **Interactive Dashboard**: Explore data, train models, and predict customer risk.  
- **Database Integration**: Connects to a PostgreSQL database.  
- **Machine Learning**: Logistic regression for credit default prediction.  
- **Explainability**: SHAP values for model interpretation.  
- **Scoring System**: Converts probability into a custom score (0–160) with risk levels.

## Project Structure

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

## Usage

1. Ensure PostgreSQL is installed and running.  
2. Update `config/config.py` with your DB credentials.  
3. Run the app:  
```bash
streamlit run app.py
````

## Risk Score Interpretation

| Score Range | Risk Level        |
| ----------- | ----------------- |
| 152–160     | 🔵 Excellent      |
| 144–151     | 🟢 Very Good      |
| 112–143     | 🟡 Medium         |
| 80–111      | 🟠 Low            |
| 0–79        | 🔴 Very High Risk |

## Project Motivation

Simulates a real-world financial scenario to evaluate customer credit default risk.
Highlights:

* Database management and connectivity
* Machine learning for classification
* Risk interpretation via scores and probabilities
* Visual and explainable results for decision makers
