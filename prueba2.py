import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import os
import pickle

st.set_page_config(page_title="Gestión de Cargas", layout="wide")

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

# --- Generar una carga aleatoria ---
def generar_carga():
    if df.empty:
        st.error("⚠️ No hay datos disponibles.")
        return None, None
    carga = df.sample(1)
    return carga.iloc[0], carga.index[0]  # Devolvemos la fila como serie y su índice

# --- Página 1: Generar Carga ---
def pagina_generar_carga():
    st.title("Generar Carga")
    if st.button("Generar Carga"):
        carga, idx = generar_carga()
        if carga is not None:
            st.session_state["carga"] = carga
            st.session_state["carga_idx"] = idx
    
    if "carga" in st.session_state:
        carga = st.session_state["carga"]
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.subheader("Detalles de la Carga")
            st.write(f"**Origen:** {carga['CityOrigin']}")
            st.write(f"**Destino:** {carga['CityDestination']}")
            st.write(f"**Peso:** {carga['Weight']} lbs")
            st.write(f"**Tamaño:** {carga['Size']} cu ft")
            
            equip = carga['Equip'].lower()
            image_path = f"images/{equip}.png"
            if os.path.exists(image_path):
                st.image(image_path, caption=equip)
            else:
                st.warning(f"Imagen no encontrada: {image_path}")
        
        with col2:
            st.subheader("Ruta en Mapa")
            mapa = folium.Map(location=[carga['LatOrigin'], carga['LngOrigin']], zoom_start=6)
            folium.Marker([carga['LatOrigin'], carga['LngOrigin']], tooltip="Origen").add_to(mapa)
            folium.Marker([carga['LatDestination'], carga['LngDestination']], tooltip="Destino").add_to(mapa)
            folium_static(mapa)
        
        with col3:
            st.subheader("Distancia Estimada")
            distancia = np.random.randint(100, 500)
            st.write(f"**Distancia:** {distancia} km")
            st.session_state["distancia"] = distancia

# --- Página 2: Vista Dueño del Vehículo ---
def pagina_dueno():
    st.title("Vista Dueño del Vehículo")
    
    # Verificar si la carga está en el estado
    if "carga" not in st.session_state or "carga_idx" not in st.session_state:
        st.warning("Genera una carga primero en la otra página.")
        return
    
    carga = st.session_state["carga"]
    idx = st.session_state["carga_idx"]
    MAPE = 0.1164  # Valor del Mape del Modelo en Validación
    if model is not None and "distancia" in st.session_state:
        if idx in df_encoded.index:
            features = df_encoded.loc[idx].values.reshape(1, -1)  # Seleccionamos la misma fila en Xtest_encoded
            pred = model.predict(features)[0]
            min_value = pred - (pred * MAPE)
            max_value = pred + (pred * MAPE)
            
            # Organizar la vista en tres columnas
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.subheader("Detalles de la Carga")
                st.write(f"**Origen:** {carga['CityOrigin']}")
                st.write(f"**Destino:** {carga['CityDestination']}")
                st.write(f"**Peso:** {carga['Weight']} lbs")
                st.write(f"**Tamaño:** {carga['Size']} cu ft")
                
                equip = carga['Equip'].lower()
                image_path = f"images/{equip}.png"
                if os.path.exists(image_path):
                    st.image(image_path, caption=equip)
                else:
                    st.warning(f"Imagen no encontrada: {image_path}")
            
            with col2:
                st.subheader("Ruta en Mapa")
                mapa = folium.Map(location=[carga['LatOrigin'], carga['LngOrigin']], zoom_start=6)
                folium.Marker([carga['LatOrigin'], carga['LngOrigin']], tooltip="Origen").add_to(mapa)
                folium.Marker([carga['LatDestination'], carga['LngDestination']], tooltip="Destino").add_to(mapa)
                folium_static(mapa)
            
            with col3:
                st.subheader("Distancia Estimada")
                distancia_km = st.session_state["distancia"]
                distancia_mi = distancia_km * 0.621371
                st.write(f"**Distancia:** {distancia_km} km ({distancia_mi:.2f} mi)")
                
                st.subheader("Estimación de Pago")
                st.write(f"💰 **Valor mínimo:** ${min_value:.2f}")
                st.write(f"💰 **Valor máximo:** ${max_value:.2f}")
        else:
            st.warning("No se encontró la fila correspondiente en Xtest_encoded.parquet.")
    else:
        st.warning("No se pudo calcular el pago. Asegúrate de que el modelo está cargado y los datos están correctamente procesados.")

# --- Menú de Navegación ---
pagina = st.sidebar.selectbox("Selecciona una página", ["Generar Carga", "Vista Dueño del Vehículo"])
if pagina == "Generar Carga":
    pagina_generar_carga()
elif pagina == "Vista Dueño del Vehículo":
    pagina_dueno()
