# eda/eda_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def correlation_heatmap(df):
    st.subheader("Korelasyon Isı Haritası")
    # 'musteri_id' gibi anlamsız ID sütunlarını kaldır
    df = df.drop(columns=['musteri_id'], errors='ignore')
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

def pie_chart_column(df, column):
    st.subheader(f"{column} için Pasta Grafiği")
    value_counts = df[column].value_counts()
    fig, ax = plt.subplots()
    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

def plot_categorical_target_distribution(df, category_column, target_column):
    st.subheader(f"{category_column} değişkenine göre ödeme durumu analizi")

    grouped = df.groupby([category_column, target_column]).size().unstack().fillna(0)
    grouped_percent = grouped.div(grouped.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped_percent.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")

    ax.set_ylabel("Yüzde (%)")
    ax.set_title(f"{category_column} kırılımında ödeme durumu\n(0 = Ödedi, 1 = Ödemedi)")
    ax.legend(title="Tahmin Sonucu", labels=["0 = Ödedi", "1 = Ödemedi"])
    st.pyplot(fig)