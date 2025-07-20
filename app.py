# app.py
import streamlit as st
import pandas as pd
from preprocessing import preprocess_pipeline, generate_visuals, save_logs

st.title("🧹 Data Preprocessing Agent")
st.write("Upload a dataset and apply full preprocessing (missing, outliers, encoding, scaling).")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
strategy = st.selectbox("Choose Missing Value Handling Strategy:", ["auto", "custom", "drop"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data")
    st.dataframe(raw_df.head())

    if st.button("Run Preprocessing"):
        df_cleaned, miss_log, outlier_log = preprocess_pipeline(raw_df, strategy)
        df_cleaned.to_csv("cleaned_output.csv", index=False)
        save_logs(miss_log, outlier_log, "logs.txt")
        generate_visuals(raw_df, df_cleaned)

        st.success("✅ Preprocessing Done!")
        st.download_button("⬇️ Download Cleaned CSV", "cleaned_output.csv", file_name="cleaned_output.csv")
        st.download_button("📝 Download Logs", "logs.txt", file_name="preprocessing_logs.txt")
        st.image("visuals/hist_before.png", caption="Histogram Before")
        st.image("visuals/hist_after.png", caption="Histogram After")
        st.image("visuals/heatmap.png", caption="Correlation Heatmap")
