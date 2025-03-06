import pandas as pd
import plotly.express as px
import streamlit as st
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

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
df = df[cols]

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

# Resumen de valores nulos por tipo de camión
summary = df.groupby('Equip')['RatePerMile'].agg(
    total='size',
    nulos=lambda x: x.isnull().sum(),
    no_nulos='count'
)
summary['% nulos por Equip'] = (summary['nulos'] / summary['total']) * 100
st.write(summary)

# Visualización: Mapa de cargas publicadas
st.header("Mapa de cargas publicadas")
if 'LatOrigin' in df.columns and 'LngOrigin' in df.columns:
    df_geo = df.dropna(subset=['LatOrigin', 'LngOrigin'])
    if not df_geo.empty:
        fig = px.scatter_geo(df_geo, lat='LatOrigin', lon='LngOrigin',
                             title="Mapa de cargas publicadas",
                             opacity=0.6)
        st.plotly_chart(fig)
    else:
        st.warning("No hay datos válidos para mostrar en el mapa.")
else:
    st.error("Las columnas LatOrigin y LngOrigin no existen en el DataFrame.")

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

# Visualización: Distribución de tarifas por milla
st.header("Distribución de tarifas por milla")
if not df.empty:
    fig = px.histogram(df, x="RatePerMile", nbins=50, title="Distribución de tarifas por milla")
    st.plotly_chart(fig)
else:
    st.warning("No hay datos válidos para mostrar en la distribución de tarifas.")

# Visualización: Boxplot de tarifas por tipo de camión
st.header("Análisis general boxplot de RatePerMile por tipo de camión")
fig = px.box(df, x="Equip", y="RatePerMile", color="Equip")
st.plotly_chart(fig)

# Visualización: Cantidad de cargas publicadas por día
st.header("Cantidad de cargas publicadas por día")
cargas_por_dia = df.groupby(df['Posted'].dt.date)['ID'].nunique().reset_index()
fig = px.bar(cargas_por_dia, x='Posted', y='ID',
             title='Cantidad de cargas publicadas por día',
             labels={'Posted': 'Día', 'ID': 'Cantidad de cargas'},
             color='ID',
             color_continuous_scale='Greys')
st.plotly_chart(fig)

# Análisis con Pandas Profiling
st.header("Análisis exploratorio con Pandas Profiling")
profile = ProfileReport(df, explorative=True)
st_profile_report(profile)

# Mapa con folium
st.header("Mapa con cargas por camión y por día")
mapa = folium.Map(location=[39.8283, -98.5795], zoom_start=5)

for _, registro in df.iterrows():
    folium.CircleMarker(
        location=[registro["LatOrigin"], registro["LngOrigin"]],
        color="blue" if registro["RatePerMile"] > 0 else "orange",
        fill=True,
    ).add_to(mapa)
st.write(mapa)

# Boxplot de RatePerMile por combinaciones de zonas
st.header("Boxplot de RatePerMile por combinaciones de zonas")
df['ZoneCombination'] = df['StateOrigin'] + '-' + df['StateDestination']
fig = px.box(df, x="ZoneCombination", y="RatePerMile", color='ZoneCombination')
st.plotly_chart(fig)
