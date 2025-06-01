
# weather_forecast_lstm_app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import sqlite3
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

DB_NAME = "weather_data.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            city TEXT,
            date TEXT,
            temperature REAL,
            humidity REAL,
            windspeed REAL,
            PRIMARY KEY (city, date)
        )
    """)
    conn.commit()
    conn.close()

def fetch_weather_data(city, lat, lon):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM weather WHERE city = ?", (city,))
    rows = cursor.fetchall()
    conn.close()

    if rows:
        df = pd.DataFrame(rows, columns=["City", "Date", "Temperature", "Humidity", "WindSpeed"])
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    else:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date=1970-01-01&end_date=2025-01-01"
            f"&daily=temperature_2m_max,relative_humidity_2m_max,windspeed_10m_max"
            f"&timezone=auto"
        )
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame({
            "City": city,
            "Date": pd.to_datetime(data["daily"]["time"]),
            "Temperature": data["daily"]["temperature_2m_max"],
            "Humidity": data["daily"]["relative_humidity_2m_max"],
            "WindSpeed": data["daily"]["windspeed_10m_max"]
        })

        # Save to DB
        conn = sqlite3.connect(DB_NAME)
        df.to_sql("weather", conn, if_exists="append", index=False)
        conn.close()
        return df

def train_or_load_model(city, feature, data):
    model_path = f"models/{city}_{feature}_model.pkl"
    scaler_path = f"models/{city}_{feature}_scaler.pkl"
    os.makedirs("models", exist_ok=True)

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))

        def create_sequences(data, seq_length=30):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data)
        model = Sequential()
        model.add(LSTM(64, input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        return model, scaler

st.title("🌦️ نظام التنبؤ بالطقس باستخدام LSTM مع SQLite Cache")

city = st.text_input("🧭 أدخل اسم المدينة (بالإنجليزية):", "Damascus")

if st.button("ابدأ التنبؤ"):
    init_db()

    with st.spinner("🔍 جارٍ تحديد الموقع..."):
        geolocator = Nominatim(user_agent="weather_forecast_app")
        location = geolocator.geocode(city)
        if not location:
            st.error("❌ لم يتم العثور على المدينة.")
            st.stop()
        lat, lon = location.latitude, location.longitude
        st.success(f"📍 {city}: {lat:.2f}, {lon:.2f}")

    with st.spinner("🌐 جارٍ تحميل البيانات أو استعادتها من SQLite..."):
        df = fetch_weather_data(city, lat, lon)
        st.success(f"✅ تم تحميل أو استرجاع {len(df)} يومًا من البيانات!")
        st.dataframe(df.tail())

    features = ["Temperature", "Humidity", "WindSpeed"]
    predictions = {}

    for feature in features:
        st.subheader(f"📊 تحليل وتنبؤ: {feature}")
        data = df[feature].dropna().values
        model, scaler = train_or_load_model(city, feature, data)

        last_sequence = scaler.transform(data[-30:].reshape(-1, 1)).reshape(1, 30, 1)
        pred_scaled = model.predict(last_sequence)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        predictions[feature] = pred

        fig, ax = plt.subplots()
        ax.plot(df["Date"], df[feature], label="Actual")
        ax.axhline(pred, color="r", linestyle="--", label="Prediction")
        ax.set_title(f"{feature} - التاريخي والتوقع")
        ax.legend()
        st.pyplot(fig)

    st.subheader("🔮 توقعات الغد:")
    for f in features:
        unit = "°C" if f == "Temperature" else "%" if f == "Humidity" else "كم/ساعة"
        st.write(f"{f}: {predictions[f]:.2f} {unit}")
