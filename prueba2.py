import streamlit as st
import pandas as pd
import numpy as np
import os, glob, ast, io
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import streamlit.components.v1 as components

st.set_page_config(page_title="Análisis de Cargas", layout="wide")

# Título y descripción
st.title("Análisis de Cargas - Proyecto Final")
st.markdown("**Equipo:** Karen Gomez, David Lizcano, Jason Barrios, Camilo Barriosnuevo")

# --- Carga y Limpieza de Datos ---
@st.cache_data
def load_data():
    folder = os.getcwd()
    files = glob.glob(os.path.join(folder, "*.parquet"))
    dfs = [pd.read_parquet(file) for file in files]
    data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    data['RatePerMile'] = pd.to_numeric(data['RatePerMile'], errors='coerce')
    cols = ['ID', 'Posted', 'CityOrigin', 'LatOrigin', 'LngOrigin', 'CityDestination',
            'LatDestination', 'LngDestination', 'Size', 'Weight', 'Distance', 'RatePerMile',
            'Equip', 'StateOrigin', 'HubOrigin', 'StateDestination', 'HubDestination']
    return data[cols]

df = load_data()
df = df.drop_duplicates('ID', keep='first')

def get_zone(state):
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
        if state in states:
            return zone
    return "Unknown"

df['ZoneOrigin'] = df['StateOrigin'].apply(get_zone)
df['ZoneDestination'] = df['StateDestination'].apply(get_zone)

def filter_and_explode_equip(data):
    desired = {'Van', 'Reefer', 'Flatbed'}
    data['Equip'] = data['Equip'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data = data[data['Equip'].apply(lambda x: isinstance(x, list))]
    data['Equip'] = data['Equip'].apply(lambda lst: [i for i in lst if i in desired])
    return data[data['Equip'].map(len) > 0].explode('Equip').reset_index(drop=True)

df = filter_and_explode_equip(df)
df['Posted'] = pd.to_datetime(df['Posted'])
df['weekday_name'] = df['Posted'].dt.day_name()
df = df[df['weekday_name'] != 'Sunday']  # Remover domingos

# --- Sidebar Resumen ---
st.sidebar.header("Resumen de Datos")
st.sidebar.write("Dimensiones:", df.shape)
st.sidebar.write("Equipos:", df['Equip'].unique())

# --- Exploración de Datos ---
st.header("Exploración de Datos")
with st.expander("Información General del DataFrame"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write("Primeras 3 filas:", df.head(3))

st.subheader("Valores Nulos")
st.write(df.isnull().sum())

fig_heat, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='inferno', ax=ax)
ax.set_title("Heatmap de Nulos")
st.pyplot(fig_heat)

# --- Mapa Interactivo ---
st.subheader("Mapa de Cargas (Origen)")
m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
for _, row in df.iterrows():
    color = "blue" if pd.notnull(row["RatePerMile"]) and row["RatePerMile"] > 0 else "orange"
    folium.CircleMarker(location=[row["LatOrigin"], row["LngOrigin"]],
                        radius=2, color=color, fill=True).add_to(m)
components.html(m._repr_html_(), height=500)

# --- Resumen por Tipo de Camión ---
st.subheader("Resumen RatePerMile por Tipo de Camión")
summary = df.groupby('Equip')['RatePerMile'].agg(
    total='size', 
    nulos=lambda x: x.isnull().sum(), 
    no_nulos='count'
)
summary['% nulos'] = (summary['nulos'] / summary['total'] * 100).map("{:.2f}%".format)
st.write(summary)
