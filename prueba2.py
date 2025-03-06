import pandas as pd
import plotly.express as px
import streamlit as st
import glob
import os
import ast

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

# Función para asignar zonas a estados
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

# Filtrar solo camiones permitidos
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

# Convertir la fecha a formato datetime
df['Posted'] = pd.to_datetime(df['Posted'])
df['weekday'] = df['Posted'].dt.weekday
df['weekday_name'] = df['Posted'].dt.day_name()

# Eliminar registros del domingo
df = df[df['weekday_name'] != 'Sunday']

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

# Visualización: Distribución de tarifas por milla
st.header("Distribución de tarifas por milla")
if 'RatePerMile' in df.columns:
    df = df.dropna(subset=['RatePerMile'])
    if not df.empty:
        fig = px.histogram(df, x="RatePerMile", nbins=50, title="Distribución de tarifas por milla")
        st.plotly_chart(fig)
    else:
        st.warning("No hay datos válidos para mostrar en la distribución de tarifas.")
else:
    st.error("La columna RatePerMile no existe en el DataFrame.")



import pandas as pd
import plotly.express as px
import streamlit as st
import glob
import os
import ast
import seaborn as sns
import matplotlib.pyplot as plt

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
