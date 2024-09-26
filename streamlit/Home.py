import streamlit as st

st.set_page_config(
    page_title="LidarDec23",
    page_icon="🗺️",
)


st.title("LIDAR_DEC_23")
st.header("Lidar2Roofs - Lidar2Trees --- ")
st.write("Détection de batiments et d'arbres depuis des photographies aériennes et des relevés Lidar. "
             "Relevé des formes sommaires de chaque instance à des fins de modélisation")

st.image("reports/Visuels/Lidar_rgb_3D.png", use_column_width=True)