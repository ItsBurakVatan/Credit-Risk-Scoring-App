# app.py
import streamlit as st
import pickle
from data.db_connector import fetch_all_data, fetch_customer_by_id, get_db_engine
from utils.preprocess import preprocess_predict, preprocess_train
from model.trainer import train_model
from model.predictor import load_model, predict_customer
from model.explainer import explain_model
from eda.eda_utils import correlation_heatmap, pie_chart_column, plot_categorical_target_distribution
from config.config import TARGET_COLUMN, TABLE_NAME
import pandas as pd
from model.predictor import predict_probability_and_score

# Sayfa yapılandırması
st.set_page_config(page_title="Kredi Skorlama Uygulaması", layout="wide")
st.title("📊 Kredi Skorlama Uygulaması")

# Menü
menu = st.sidebar.radio("Menü", ["Veri Analizi", "Model Eğitimi", "Tahmin", "Müşteri Analizi"])

# Veritabanı bağlantısı
engine = get_db_engine()
df = fetch_all_data()

# Menü Seçimi
if menu == "Veri Analizi":
    st.dataframe(df.head())
    correlation_heatmap(df)

    excluded_columns = ['musteri_id', 'isim']
    categorical_columns = [col for col in df.columns if col not in excluded_columns and df[col].dtype == 'object']

    if categorical_columns:
        selected_column = st.selectbox("Pasta Grafiği için sütun seçin", categorical_columns)
        pie_chart_column(df, selected_column)
    else:
        st.warning("Pasta grafiği oluşturulacak uygun kategorik sütun yok.")

elif menu == "Model Eğitimi":
    X, y, scaler, numeric_cols = preprocess_train(df, TARGET_COLUMN)
    model, report = train_model(X, y, scaler)

    with open('model/saved_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('model/numeric_cols.pkl', 'wb') as f:
        pickle.dump(numeric_cols, f)

    st.success("Model eğitildi ve kaydedildi.")
    st.json(report)

elif menu == "Tahmin":
    customer_id = st.number_input("Müşteri ID", min_value=0, step=1)
    cust_df = fetch_customer_by_id(engine, TABLE_NAME, customer_id)

    if cust_df is not None and not cust_df.empty:
        st.write(cust_df)

        if st.button("Tahmin Et"):
            def convert_to_numeric(val):
                try:
                    return float(val)
                except:
                    return 0

            cust_df['kredi_karti_odeme_performansi'] = cust_df['kredi_karti_odeme_performansi'].apply(convert_to_numeric)

            model = load_model()
            with open('model/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('model/numeric_cols.pkl', 'rb') as f:
                numeric_cols = pickle.load(f)

            X = preprocess_predict(cust_df, scaler, numeric_cols)
            pred, prob = predict_customer(model, X)
            risk_score = prob[0][1]

            st.metric("Tahmin", "✅ Ödeyecek" if pred[0] == 0 else "❌ Ödeyemeyecek")
            st.metric("Ödememe Olasılığı", f"{risk_score * 100:.2f}%")

            # Yeni eklenen puan hesaplama bölümü
            score, score_label = predict_probability_and_score(risk_score)
            st.metric("Kredi Puanı", f"{score} Puan")
            st.write(f"**Puan Değerlendirmesi:** {score_label}")

            # Risk kategorisi
            if risk_score <= 0.05:
                risk_label = "🔵 Mükemmel Müşteri – Batma riski yok"
            elif risk_score <= 0.20:
                risk_label = "🟢 Düşük Riskli Müşteri"
            elif risk_score <= 0.30:
                risk_label = "🟡 Orta Riskli Müşteri"
            elif risk_score <= 0.50:
                risk_label = "🟠 Yüksek Riskli Müşteri"
            else:
                risk_label = "🔴 Çok Yüksek Riskli Müşteri"

            st.write(f"**Model Bazlı Risk Değerlendirmesi:** {risk_label}")

            # Bonus: Ağırlıklar
            st.subheader("📊 Modelin Öğrendiği Ağırlıklar")
            weights_df = pd.DataFrame({
                "Özellik": numeric_cols,
                "Ağırlık (Katsayı)": model.coef_[0]
            }).sort_values(by="Ağırlık (Katsayı)", ascending=False)
            st.dataframe(weights_df)
            st.caption(f"Intercept (bias): {model.intercept_[0]:.4f}")
    else:
        st.warning("Müşteri bulunamadı.")

elif menu == "Müşteri Analizi":
    st.subheader("🔎 Kategorik Değişkenlere Göre Ödeme Performansı")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['isim', 'ad', 'soyad', 'id', TARGET_COLUMN]]

    if categorical_cols:
        selected_col = st.selectbox("Analiz için bir sütun seçin", categorical_cols)
        plot_categorical_target_distribution(df, selected_col, TARGET_COLUMN)
    else:
        st.info("Uygun kategorik sütun bulunamadı.")