import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and dataset
model = joblib.load("rf_model.pkl")
file_path = "final_dataset.csv"
data = pd.read_csv(file_path)

# Preprocessing
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['HomeTeam'] = label_encoder.fit_transform(data['HomeTeam'])
data['AwayTeam'] = label_encoder.fit_transform(data['AwayTeam'])

# Streamlit App
st.title("Football Match Result Prediction")
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Dataset", "Visualization", "Prediction"])

if menu == "Home":
    st.write("## Selamat Datang di Aplikasi Prediksi Sepakbola ")
    st.write(
        """
        Aplikasi ini memprediksi hasil dari pertandingan sepakbola berdasarkan hasil pertandingan sebelum - sebelumnya.
        """
    )

elif menu == "Dataset":
    st.write("## Dataset Overview")
    st.dataframe(data)

elif menu == "Visualization":
    st.write("## Data Visualization")
    st.bar_chart(data['FTR'].value_counts())
    st.write("### Top Home Teams")
    top_home_teams = data.groupby('HomeTeam').size().sort_values(ascending=False).head(10)
    st.bar_chart(top_home_teams)

elif menu == "Prediction":
    st.write("## Make a Prediction")

    # User Input
    home_team = st.selectbox("Select Home Team:", label_encoder.classes_)
    away_team = st.selectbox("Select Away Team:", label_encoder.classes_)
    htgs = st.number_input("Home Team Goals Scored:", min_value=0, max_value=10, value=0)
    atgs = st.number_input("Away Team Goals Scored:", min_value=0, max_value=10, value=0)
    htgd = st.number_input("Home Team Goal Difference:", min_value=-10, max_value=10, value=0)
    atgd = st.number_input("Away Team Goal Difference:", min_value=-10, max_value=10, value=0)

    # Prediction
    if st.button("Predict"):
        input_data = pd.DataFrame(
            [[
                label_encoder.transform([home_team])[0],
                label_encoder.transform([away_team])[0],
                htgs, atgs, htgd, atgd
            ]],
            columns=['HomeTeam', 'AwayTeam', 'HTGS', 'ATGS', 'HTGD', 'ATGD']
        )
        prediction = model.predict(input_data)[0]
        result = ["Home Win", "Draw", "Away Win"][prediction]
        st.write(f"### Predicted Result: {result}")
