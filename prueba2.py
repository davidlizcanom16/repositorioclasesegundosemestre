import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import os
import pickle

st.set_page_config(page_title="Gestión de Cargas", layout="wide")

# --- Pestaña 1: Introducción ---
def pagina_introduccion():
    st.title("Introducción")
    st.write("Esta aplicación ha sido desarrollada en Google Colab y desplegada usando Streamlit.")
    st.write("Proporciona una herramienta para la gestión de cargas en logística, permitiendo visualizar rutas, estimar costos y analizar datos de transporte.")
    st.write("### Funcionalidades:")
    st.write("- Generación de carga aleatoria con detalles del origen y destino.")
    st.write("- Visualización de la ruta en un mapa interactivo.")
    st.write("- Cálculo del costo estimado de transporte utilizando un modelo de Machine Learning.")

# --- Pestaña 2: Datos Utilizados ---
def pagina_datos():
    st.title("Datos Utilizados")
    st.write("Esta aplicación utiliza dos conjuntos de datos principales:")
    st.write("1. **dataset.parquet**: Contiene la información detallada de las cargas.")
    st.write("2. **Xtest_encoded.parquet**: Contiene la versión codificada de los datos utilizada para hacer predicciones.")
    st.write("El modelo de predicción ha sido entrenado previamente y guardado como **random_forest_model.pkl**.")

# --- Pestaña 3: Modelo de Predicción ---
def pagina_modelo():
    st.title("Modelo de Predicción")
    st.write("El modelo utilizado en esta aplicación es un Random Forest Regressor entrenado para estimar los costos de transporte.")
    st.write("Se ha calculado un **Mean Absolute Percentage Error (MAPE)** de **11.64%**, lo que indica un buen desempeño en la estimación de costos.")
    st.write("El modelo ha sido entrenado con datos reales de transporte y utiliza variables como el tipo de vehículo, la distancia y el peso para hacer las predicciones.")

# --- Cargar datos y modelo ---
@st.cache_data
def load_data():
    file_path = "dataset.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        st.error("⚠️ No se encontró el archivo dataset.parquet")
        return pd.DataFrame()

df = load_data()

@st.cache_data
def load_encoded_data():
    file_path = "Xtest_encoded.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        st.error("⚠️ No se encontró el archivo Xtest_encoded.parquet")
        return pd.DataFrame()

df_encoded = load_encoded_data()

@st.cache_data
def load_model():
    model_path = "random_forest_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)
    else:
        st.error("⚠️ No se encontró el archivo random_forest_model.pkl")
        return None

model = load_model()

# --- Menú de Navegación ---
pagina = st.sidebar.selectbox("Selecciona una página", ["Introducción", "Datos Utilizados", "Modelo de Predicción", "Generar Carga", "Vista Dueño del Vehículo"])
if pagina == "Introducción":
    pagina_introduccion()
elif pagina == "Datos Utilizados":
    pagina_datos()
elif pagina == "Modelo de Predicción":
    pagina_modelo()
elif pagina == "Generar Carga":
    pagina_generar_carga()
elif pagina == "Vista Dueño del Vehículo":
    pagina_dueno()
