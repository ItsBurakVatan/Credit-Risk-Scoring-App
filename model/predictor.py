# model/predictor.py
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import streamlit as st
from config.config import MODEL_PATH

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def predict_customer(model, df):
    return model.predict(df), model.predict_proba(df)

def plot_roc_pr(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Precision-Recall Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(recall, precision)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    st.pyplot(fig2)

def predict_probability_and_score(prob):
    """
    Olasılığa göre kredi puanı hesaplar.
    Args:
        prob (float): 1'e yakınsa risk yüksek demektir (ödememe olasılığı)
    Returns:
        int: 0-160 arasında bir kredi puanı
    """
    # Başlangıç puanı: 160 (en güvenli müşteri)
    # Olasılık arttıkça puan düşer
    score = round((1 - prob) * 160)

    # Aralıkları isteğe göre uyarlayabilirsin
    if score >= 152:
        label = "🔵 Mükemmel Puan"
    elif 144 <= score < 152:
        label = "🟢 Çok İyi Puan"
    elif 112 <= score < 144:
        label = "🟡 Orta Puan"
    elif 80 <= score < 112:
        label = "🟠 Düşük Puan"
    else:
        label = "🔴 Kötü Puan"

    return score, label
