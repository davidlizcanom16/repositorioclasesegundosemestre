import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import ast
import plotly.express as px

# Título y descripción del proyecto
st.title("Proyecto Final Modelos Analíticos - Maestría Analítica Datos")
st.markdown("""
**Miembros del Equipo:**  
- Karen Gomez  
- David Lizcano  
- Jason Barrios  
- Camilo Barriosnuevo  
""")

st.markdown("""
**Contexto de Negocio: Situación problema**

Una empresa norteamericana dedicada a conectar generadores de carga y transportistas necesita predecir la tarifa (`RatePerMile`) que un cliente pagará por carga transportada, 
dada la dinámica de mercado (oferta/demanda). El objetivo es suministrar información oportuna a los transportistas mostrando las ofertas de carga por zona, día, cliente y estimación de tarifas.
""")

# Función para cargar datos (se usa @st.cache para acelerar recargas)
@st.cache
def load_data():
    folder_base = os.getcwd()
    parquet_files = glob.glob(os.path.join(folder_base, "*.parquet"))
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)
    if dfs:
        data = pd.concat(dfs, ignore_index=True)
    else:
        data = pd.DataFrame()
    # Asegurarse de que RatePerMile sea numérico
    data['RatePerMile'] = pd.to_numeric(data['RatePerMile'], errors='coerce')
    cols = ['ID', 'Posted', 'CityOrigin', 'LatOrigin', 'LngOrigin', 'CityDestination',
            'LatDestination', 'LngDestination', 'Size', 'Weight', 'Distance', 'RatePerMile',
            'Equip', 'StateOrigin', 'HubOrigin', 'StateDestination', 'HubDestination']
    data = data[cols]
    return data

df = load_data()
st.write("Dimensiones del dataset:", df.shape)

# Eliminación de duplicados
st.write("Duplicados a eliminar:", df.shape[0] - df['ID'].nunique())
df = df.drop_duplicates('ID', keep='first')
st.write("Nuevas dimensiones después de eliminar duplicados:", df.shape)

# Función para asignar zona según el estado
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

# Filtrado y tratamiento de la columna 'Equip'
def filter_and_explode_equip(data):
    desired_values = ['Van', 'Reefer', 'Flatbed']
    column_name = 'Equip'
    desired_values_set = set(desired_values)
    
    # Convertir cadena a lista si es necesario
    data[column_name] = data[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data = data[data[column_name].apply(lambda x: isinstance(x, list))]
    data[column_name] = data[column_name].apply(lambda x: [item for item in x if item in desired_values_set])
    data = data[data[column_name].map(len) > 0]
    data = data.explode(column_name).reset_index(drop=True)
    return data

df = filter_and_explode_equip(df)
st.write("Dimensiones después de filtrar equipos:", df.shape)
st.write("Equipos disponibles:", df['Equip'].unique())

# Extracción del día de la semana a partir de la columna 'Posted'
df['Posted'] = pd.to_datetime(df['Posted'])
df['weekday'] = df['Posted'].dt.weekday
df['weekday_name'] = df['Posted'].dt.day_name()

weekday_pivot = df.pivot_table(index='weekday_name', values='ID', aggfunc='count')
st.write("Número de cargas por día de la semana:", weekday_pivot)

# Eliminación de registros de domingo (si existen)
if 'Sunday' in df['weekday_name'].unique():
    df = df.drop(df.loc[df['weekday_name'] == 'Sunday'].index)
    st.write("Dimensiones después de eliminar domingos:", df.shape)

st.success("Proceso de carga y limpieza completado.")

import io
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import streamlit.components.v1 as components

st.header("2. Análisis Exploratorio de Datos 1")

# Dataframe inicial: Información general y primeras filas
st.subheader("Información general del DataFrame")
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

st.subheader("Primeras 3 filas")
st.write(df.head(3))

# Análisis de valores nulos
st.markdown("### Análisis de Valores Nulos")
null_counts = df.isnull().sum()
st.write("Cantidad de valores nulos por columna:")
st.write(null_counts)

st.markdown("**Heatmap de Valores Nulos**")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='inferno', ax=ax)
ax.set_title('Heatmap de Valores Nulos')
ax.set_xlabel('Columnas')
ax.set_ylabel('Filas')
st.pyplot(fig)

st.markdown("**Histograma de valores nulos de RatePerMile por Equip**")
filtered_df = df[df['RatePerMile'].isnull()]
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.histplot(data=filtered_df, x='Equip', hue='Equip', ax=ax2)
ax2.set_title('Histograma de valores nulos de RatePerMile por Equip')
ax2.set_xlabel('Equip')
ax2.set_ylabel('Cantidad de valores nulos')
st.pyplot(fig2)

# Agrupar por 'Equip' y calcular estadísticas de nulos
st.markdown("**Resumen de nulos por tipo de camión (Equip)**")
summary = df.groupby('Equip')['RatePerMile'].agg(
    total='size',
    nulos=lambda x: x.isnull().sum(),
    no_nulos='count'
)
summary['% nulos por Equip'] = (summary['nulos'] / summary['total']) * 100
summary['% nulos por Equip'] = summary['% nulos por Equip'].map("{:.2f}%".format)
st.write(summary)

st.markdown("**Descripción estadística del DataFrame**")
st.write(df.describe())

# Mapa: Cargas con y sin tarifa publicada por Estado
st.header("Mapa Situación Actual: Cargas con y sin tarifa publicada por Estado")

# Agrupación por estado: cálculo de envíos totales y nulos
state_total_counts    = df.groupby('StateOrigin')['RatePerMile'].size()
state_non_null_counts = df.groupby('StateOrigin')['RatePerMile'].count()
state_null_counts     = df[df['RatePerMile'].isnull()].groupby('StateOrigin').size()

summary_df = pd.DataFrame({
    'Envíos sin Rate': state_null_counts,
    'Envíos con Rate': state_non_null_counts,
    'Total_Envíos': state_total_counts
})
summary_df = summary_df.fillna(0).astype(int)
summary_df['% Envíos null'] = (summary_df['Envíos sin Rate'] / summary_df['Total_Envíos']) * 100
summary_df['% Envíos null'] = summary_df['% Envíos null'].map("{:.2f}%".format)
summary_df = summary_df.sort_values(by=['Total_Envíos'], ascending=False)
st.write(summary_df)

st.markdown("**Mapa interactivo de cargas según RatePerMile**")
# Crear el mapa centrado en Estados Unidos
m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)

# Se añaden marcadores: azul si existe tarifa positiva, naranja si es nula o no positiva
for _, row in df.iterrows():
    color = "blue" if pd.notnull(row["RatePerMile"]) and row["RatePerMile"] > 0 else "orange"
    folium.CircleMarker(
        location=[row["LatOrigin"], row["LngOrigin"]],
        radius=3,
        color=color,
        fill=True,
    ).add_to(m)

# Mostrar el mapa en Streamlit
components.html(m._repr_html_(), height=600)
