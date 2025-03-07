import streamlit as st
import pandas as pd
import glob
import os
import ast
import folium
from streamlit_folium import folium_static

# Configuración de la aplicación
st.set_page_config(page_title="Predicción de Tarifas de Carga", layout="wide")
st.title("📦 Predicción de Tarifas de Carga")

# Función para cargar datos
@st.cache_data
def load_data():
    folder_base = os.getcwd()
    parquet_files = glob.glob(os.path.join(folder_base, "*.parquet"))
    
    if not parquet_files:
        st.error("No se encontraron archivos Parquet en la carpeta actual.")
        return pd.DataFrame()  # Retornamos un DataFrame vacío
    
    dfs = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df['RatePerMile'] = pd.to_numeric(df['RatePerMile'], errors='coerce')
    
    cols = ['ID', 'Posted', 'CityOrigin', 'LatOrigin', 'LngOrigin', 'CityDestination',
            'LatDestination', 'LngDestination', 'Size', 'Weight', 'Distance', 'RatePerMile',
            'Equip', 'StateOrigin', 'HubOrigin', 'StateDestination', 'HubDestination']
    df = df[cols]
    
    return df

df = load_data()

st.write("### Datos cargados:")
if not df.empty:
    st.dataframe(df.head())
else:
    st.write("No hay datos para mostrar.")

# Eliminación de duplicados
if not df.empty:
    df = df.drop_duplicates('ID', keep='first')
    st.write(f"### Registros después de eliminar duplicados: {df.shape[0]}")

# Función para asignar zonas
@st.cache_data
def get_zone(state_code):
    zones = {
        "Z0": {"CT", "ME", "MA", "NH", "NJ", "RI", "VT"},
        "Z1": {"DE", "NY", "PA"},
        "Z2": {"MD", "NC", "SC", "VA", "WV"},
        "Z3": {"AL", "FL", "GA", "MS", "TN"},
        "Z4": {"IN", "KY", "MI", "OH"},
        "Z5": {"IA", "MN", "MT", "ND", "SD", "WI"},
        "Z6": {"IL", "KS", "MO", "NE"},
        "Z7": {"AR", "LA", "OK", "TX"},
        "Z8": {"AZ", "CO", "ID", "NV", "NM", "UT", "WY"},
        "Z9": {"CA", "OR", "WA", "AK"}
    }
    for zone, states in zones.items():
        if state_code in states:
            return zone
    return "Unknown"

if not df.empty:
    df['ZoneOrigin'] = df['StateOrigin'].apply(get_zone)
    df['ZoneDestination'] = df['StateDestination'].apply(get_zone)

# Filtro de camiones
def filter_and_explode_equip(data):
    desired_values = ['Van', 'Reefer', 'Flatbed']
    column_name = 'Equip'
    data[column_name] = data[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data = data[data[column_name].apply(lambda x: isinstance(x, list))]    
    data[column_name] = data[column_name].apply(lambda x: [item for item in x if item in desired_values])
    data = data[data[column_name].map(len) > 0]
    return data.explode(column_name).reset_index(drop=True)

if not df.empty:
    df = filter_and_explode_equip(df)
    st.write(f"### Registros después de filtrar camiones: {df.shape[0]}")

# Función para mostrar el mapa
def plot_map(data):
    # Filtrar filas con coordenadas válidas
    data = data.dropna(subset=['LatOrigin', 'LngOrigin'])
    
    if data.empty:
        return folium.Map(location=[39.8283, -98.5795], zoom_start=5)
    
    # Centrar el mapa en el promedio de coordenadas
    center_lat = data['LatOrigin'].mean()
    center_lng = data['LngOrigin'].mean()
    m = folium.Map(location=[center_lat, center_lng], zoom_start=5)
    
    for _, row in data.iterrows():
        try:
            lat = float(row['LatOrigin'])
            lng = float(row['LngOrigin'])
        except (TypeError, ValueError):
            continue
        
        folium.Marker(
            location=[lat, lng],
            popup=row['CityOrigin'],
            icon=folium.Icon(color='blue', icon='cloud')
        ).add_to(m)
    return m

st.write("### Mapa de Ubicaciones de Origen")
if not df.empty:
    folium_static(plot_map(df))
else:
    st.write("No hay datos para mostrar en el mapa.")
