import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# Cargar archivos Parquet y consolidar datos
def load_data():
    folder_base = os.getcwd()
    parquet_files = glob.glob(folder_base + "/*.parquet")
    dfs = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df['RatePerMile'] = pd.to_numeric(df['RatePerMile'], errors='coerce')
    return df

df = load_data()

# Selección de columnas clave
cols = ['ID', 'Posted', 'CityOrigin', 'StateOrigin', 'CityDestination', 'StateDestination', 'Distance', 'RatePerMile', 'Equip']
df = df[cols]

# Eliminar duplicados
df = df.drop_duplicates('ID', keep='first')

# Convertir fechas y extraer día de la semana
df['Posted'] = pd.to_datetime(df['Posted'])
df['weekday'] = df['Posted'].dt.weekday
df = df[df['weekday'] != 6]  # Eliminar domingos

# Función para asignar zonas a los estados
def get_zone(state_code):
    zones = {
        "Noreste": {"CT", "ME", "MA", "NH", "NJ", "RI", "VT", "DE", "NY", "PA"},
        "Sureste": {"MD", "NC", "SC", "VA", "WV", "AL", "FL", "GA", "MS", "TN"},
        "Centro": {"IN", "KY", "MI", "OH", "IA", "MN", "MT", "ND", "SD", "WI", "IL", "KS", "MO", "NE"},
        "Sur": {"AR", "LA", "OK", "TX"},
        "Oeste": {"AZ", "CO", "ID", "NV", "NM", "UT", "WY", "CA", "OR", "WA", "AK"}
    }
    for zone, states in zones.items():
        if state_code in states:
            return zone
    return "Desconocido"

df['ZoneOrigin'] = df['StateOrigin'].apply(get_zone)
df['ZoneDestination'] = df['StateDestination'].apply(get_zone)

# Filtrar por tipo de camión
df = df[df['Equip'].isin(['Van', 'Reefer', 'Flatbed'])]

# Visualización: Distribución de tarifas
st.subheader("Distribución de tarifas por milla")
fig = px.histogram(df, x='RatePerMile', nbins=50, title='Distribución de tarifas por milla')
st.plotly_chart(fig)

# Visualización: Mapa de cargas
st.subheader("Mapa de cargas publicadas")
fig = px.scatter_geo(df, lat='LatOrigin', lon='LngOrigin',
                     hover_name='CityOrigin',
                     title="Ubicación de cargas publicadas",
                     projection="natural earth")
st.plotly_chart(fig)

# Visualización: Tarifas por zona
st.subheader("Tarifas promedio por zona")
avg_rates = df.groupby('ZoneOrigin')['RatePerMile'].mean().reset_index()
fig = px.bar(avg_rates, x='ZoneOrigin', y='RatePerMile', title="Tarifas promedio por zona")
st.plotly_chart(fig)

st.write("Datos procesados y visualizaciones generadas correctamente.")
