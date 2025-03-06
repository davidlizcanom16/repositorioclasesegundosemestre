import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import sweetviz
import glob
import os
import ast

# --- TÍTULO Y DESCRIPCIÓN DEL PROYECTO ---
st.markdown("""
# **Proyecto Final Modelos Analíticos - Maestría Analítica Datos**
### **Miembros del Equipo:**
- Karen Gomez
- David Lizcano
- Jason Barrios
- Camilo Barriosnuevo
""")

st.markdown("""
## Contexto de Negocio: Situación problema
Empresa de bandera Norteamericana dedicada a conectar generadores de carga y transportistas con las necesidades de envío  a través de sus servicios tecnológicos. Se busca predecir cuánto pagará un determinado cliente por carga transportada.

El objetivo es suministrar información oportuna a los transportistas, brindando visibilidad de las ofertas de carga dependiendo de la zona, día, cliente y estimación de las tarifas. Se entrenará un modelo para encontrar la variable `RatePerMile`, que anticipará la tarifa que ofertará un cliente y se comparará con la de mercado.
""")

# --- CARGA DE DATOS ---
st.header("Carga de Datos")
df = pd.DataFrame()
folder_base = os.getcwd()
parquet_files = glob.glob(folder_base + "/*.parquet")

if parquet_files:
    dfs = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df['RatePerMile'] = pd.to_numeric(df['RatePerMile'], errors='coerce')
    cols = ['ID', 'Posted', 'CityOrigin', 'LatOrigin', 'LngOrigin', 'CityDestination',
            'LatDestination', 'LngDestination', 'Size', 'Weight', 'Distance', 'RatePerMile',
            'Equip', 'StateOrigin', 'HubOrigin', 'StateDestination', 'HubDestination']
    df = df[cols]
    df.to_parquet("loads.parquet")
    st.write("Datos cargados con éxito:", df.shape)
else:
    st.error("No se encontraron archivos .parquet en la carpeta.")

# --- ELIMINACIÓN DE DUPLICADOS ---
st.subheader("Eliminación de duplicados")
duplicates = 18625 - df['ID'].nunique()
st.write(f"Se encontraron {duplicates} duplicados, serán eliminados.")
df = df.drop_duplicates('ID', keep='first')
st.write("Nuevas dimensiones del dataset:", df.shape)

# --- ZONAS USA ---
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

df['ZoneOrigin'] = df['StateOrigin'].apply(get_zone)
df['ZoneDestination'] = df['StateDestination'].apply(get_zone)
st.write("Zonas de origen únicas:", df['ZoneOrigin'].unique())

# --- FILTRAR TIPOS DE CAMIONES ---
st.subheader("Filtrado de Camiones")
def filter_and_explode_equip(data):
    desired_values = ['Van', 'Reefer', 'Flatbed']
    column_name = 'Equip'
    desired_values_set = set(desired_values)

    data[column_name] = data[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data = data[data[column_name].apply(lambda x: isinstance(x, list))]
    data[column_name] = data[column_name].apply(lambda x: [item for item in x if item in desired_values_set])
    data = data[data[column_name].map(len) > 0]
    return data.explode(column_name).reset_index(drop=True)

df = filter_and_explode_equip(df)
st.write("Equipos disponibles tras filtrado:", df['Equip'].unique())

# --- DÍA DE LA SEMANA ---
df['Posted'] = pd.to_datetime(df['Posted'])
df['weekday'] = df['Posted'].dt.weekday
df['weekday_name'] = df['Posted'].dt.day_name()
st.write("Distribución de cargas por día de la semana:")
st.write(df.pivot_table(index='weekday_name', values='ID', aggfunc='count'))

# --- ELIMINACIÓN DE DOMINGOS ---
df = df[df['weekday_name'] != 'Sunday']
st.write("Dimensiones después de eliminar domingos:", df.shape)
