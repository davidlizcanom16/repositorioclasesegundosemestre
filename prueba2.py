import pandas as pd
import plotly.express as px
import streamlit as st
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import numpy as np

# Título de la aplicación
st.title("Análisis de Cargas Publicadas")

# Cargar los datos
st.header("Cargando datos")
folder_base = os.getcwd()
parquet_files = glob.glob(folder_base + "/*.parquet")

dfs = []
for file in parquet_files:
    df = pd.read_parquet(file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Convertir RatePerMile a numérico
df['RatePerMile'] = pd.to_numeric(df['RatePerMile'], errors='coerce')

# Selección de columnas importantes
cols = ['ID', 'Posted', 'CityOrigin', 'LatOrigin', 'LngOrigin', 'CityDestination',
        'LatDestination', 'LngDestination', 'Size', 'Weight', 'Distance', 'RatePerMile',
        'Equip', 'StateOrigin', 'HubOrigin', 'StateDestination', 'HubDestination']
df = df[[col for col in cols if col in df.columns]]

# Eliminar duplicados por ID
df = df.drop_duplicates('ID', keep='first')

# Convertir la fecha a formato datetime
df['Posted'] = pd.to_datetime(df['Posted'])
df['weekday'] = df['Posted'].dt.weekday
df['weekday_name'] = df['Posted'].dt.day_name()

# Eliminar registros del domingo
df = df[df['weekday_name'] != 'Sunday']

# Análisis de valores nulos
st.header("Análisis de valores nulos")
st.write("Cantidad de valores nulos por columna:")
st.write(df.isnull().sum())

plt.figure(figsize=(8, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='inferno')
st.pyplot(plt)

# Eliminación de valores nulos en RatePerMile
df = df.dropna(subset=['RatePerMile'])

# Tratamiento de outliers
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
    return df[mask]

df = remove_outliers_iqr(df, 'RatePerMile')

# Análisis de correlaciones
st.header("Análisis de correlaciones")

# Correlaciones numéricas
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
correlation_matrix = df[numeric_columns].corr()
st.write("Matriz de correlaciones:")
st.write(correlation_matrix)

# Gráfico de correlaciones
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(plt)

# Mapa con cargas por camión y por día
st.header("Mapa de Cargas Publicadas")
if 'LatOrigin' in df.columns and 'LngOrigin' in df.columns:
    m = folium.Map(location=[df['LatOrigin'].mean(), df['LngOrigin'].mean()], zoom_start=6)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['LatOrigin'], row['LngOrigin']],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m)
    st_folium(m, width=700, height=500)

# Visualización de correlaciones con gráficos interactivos
fig = px.scatter_matrix(df, dimensions=numeric_columns, title="Matriz de Dispersión de Variables Numéricas")
st.plotly_chart(fig)
