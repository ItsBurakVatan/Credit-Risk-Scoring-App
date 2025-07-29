# model/explainer.py
import shap
import streamlit as st
import matplotlib.pyplot as plt

def explain_model(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    st.subheader("SHAP Önem Grafiği")

    # Matplotlib ile uyumlu hale getirmek için shap summary plotu doğrudan çizdir
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False, ax=ax)
    st.pyplot(fig)
