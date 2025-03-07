import streamlit as st

# Configuración de la página (DEBE SER LA PRIMERA LÍNEA DE STREAMLIT)
st.set_page_config(page_title="Predicción de Tarifas de Carga", layout="wide")

# Resto de las importaciones
import pandas as pd
import glob
import os
import ast
import folium
import io
import matplotlib.pyplot
import seaborn as sns
from streamlit_folium import folium_static

# Título de la aplicación
st.title("📦 Predicción de Tarifas de Carga")

######################
# Sección 1: Carga y Limpieza de Datos, Mapa de Ubicaciones de Origen
######################

@st.cache_data
def load_data():
    folder_base = os.getcwd()
    parquet_files = glob.glob(os.path.join(folder_base, "*.parquet"))
    
    if not parquet_files:
        st.error("No se encontraron archivos Parquet en la carpeta actual.")
        return pd.DataFrame()
    
    dfs = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df['RatePerMile'] = pd.to_numeric(df['RatePerMile'], errors='coerce')
    
    cols = ['ID', 'Posted', 'CityOrigin', 'LatOrigin', 'LngOrigin', 'CityDestination',
            'LatDestination', 'LngDestination', 'Size', 'Weight', 'Distance', 'RatePerMile',
            'Equip', 'StateOrigin', 'HubOrigin', 'StateDestination', 'HubDestination']
    df = df[cols]
    
    return df

df = load_data()

st.write("### Datos Cargados:")
if not df.empty:
    st.dataframe(df.head())
else:
    st.write("No hay datos para mostrar.")

if not df.empty:
    df = df.drop_duplicates('ID', keep='first')
    st.write(f"### Registros después de eliminar duplicados: {df.shape[0]}")

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

def plot_map(data):
    data = data.dropna(subset=['LatOrigin', 'LngOrigin'])
    if data.empty:
        return folium.Map(location=[39.8283, -98.5795], zoom_start=5)
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

######################
# Sección 2: Análisis Exploratorio de Datos 1
######################
st.header("Análisis Exploratorio de Datos 1")

buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text("Información del DataFrame:")
st.text(info_str)

st.write("Primeras 3 filas del DataFrame:")
st.dataframe(df.head(3))

null_counts = df.isnull().sum()
st.write("Cantidad de valores nulos por columna:")
st.write(null_counts)

fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='inferno', ax=ax_heat)
ax_heat.set_title('Heatmap de Valores Nulos')
ax_heat.set_xlabel('Columnas')
ax_heat.set_ylabel('Filas')
st.pyplot(fig_heat)

filtered_df = df[df['RatePerMile'].isnull()]
fig_hist, ax_hist = plt.subplots()
sns.histplot(data=filtered_df, x='Equip', hue='Equip', ax=ax_hist)
ax_hist.set_title('Histograma de valores nulos de RatePerMile por Equip')
ax_hist.set_xlabel('Equip')
ax_hist.set_ylabel('Cantidad de valores nulos')
st.pyplot(fig_hist)

summary = df.groupby('Equip')['RatePerMile'].agg(
    total='size',
    nulos=lambda x: x.isnull().sum(),
    no_nulos='count'
)
summary['% nulos por Equip'] = (summary['nulos'] / summary['total']) * 100
summary['% nulos por Equip'] = summary['% nulos por Equip'].map("{:.2f}%".format)
st.write("Estadísticas de RatePerMile por Equip:")
st.dataframe(summary)

st.write("Descripción del DataFrame:")
st.write(df.describe())

st.header("Mapa: Situación Actual - Cargas con y sin tarifa publicada por Estado")

state_total_counts = df.groupby('StateOrigin')['RatePerMile'].size()
state_non_null_counts = df.groupby('StateOrigin')['RatePerMile'].count()
state_null_counts = df[df['RatePerMile'].isnull()].groupby('StateOrigin').size()

summary_df = pd.DataFrame({
    'Envíos sin Rate': state_null_counts,
    'Envíos con Rate': state_non_null_counts,
    'Total_Envíos': state_total_counts
})
summary_df = summary_df.fillna(0).astype(int)
summary_df['% Envíos null'] = (summary_df['Envíos sin Rate'] / summary_df['Total_Envíos']) * 100
summary_df['% Envíos null'] = summary_df['% Envíos null'].map("{:.2f}%".format)
summary_df = summary_df.sort_values(by=['Total_Envíos'], ascending=False)

st.write("Resumen de envíos por Estado:")
st.dataframe(summary_df)

m2 = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
for _, row in df.iterrows():
    if pd.notnull(row["LatOrigin"]) and pd.notnull(row["LngOrigin"]):
        color = "blue" if (pd.notnull(row["RatePerMile"]) and row["RatePerMile"] > 0) else "orange"
        folium.CircleMarker(
            location=[row["LatOrigin"], row["LngOrigin"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m2)

st.write("Mapa: Cargas con y sin tarifa publicada por Estado")
folium_static(m2)
