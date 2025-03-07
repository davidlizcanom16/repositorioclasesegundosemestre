import streamlit as st
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import ast
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm


# Nota: En lugar de los comandos !pip install, asegúrate de incluir las dependencias
# en un archivo requirements.txt para cuando despliegues la aplicación en Streamlit.

# Configuración de la página
st.set_page_config(page_title="Proyecto Final Modelos Analíticos", layout="wide")

# Título y descripción del proyecto
st.title("Proyecto Final Modelos Analíticos - Maestría Analítica Datos")
st.markdown("""
### Miembros del Equipo:
- Karen Gomez
- David Lizcano
- Jason Barrios
- Camilo Barriosnuevo
""")

st.markdown("## Contexto de Negocio: Situación problema")
st.markdown("""
Empresa de bandera norteamericana dedicada a conectar generadores de carga y transportistas con las necesidades de envío a través de sus servicios tecnológicos, se encuentra en la búsqueda de predecir cuánto va a pagar un determinado cliente por carga transportada. Debido a la dinámica del mercado (oferta/demanda), algunos clientes no suelen publicar las tarifas de envío en el portal, generando incertidumbre en las condiciones de negociación con los transportistas.

El objetivo es suministrar información oportuna a los transportistas dando visibilidad a las ofertas de carga dependiendo de la zona, día, cliente y estimación de tarifas. Para ello se entrenará un modelo que estime la variable `RatePerMile` (tarifa por milla) y se comparará con la tarifa de mercado.
""")

st.markdown("## 1. Limpieza de información")
st.markdown("### 1.1 Cargas Broker")
st.markdown("""
Acciones:
- Filtrar broker.
- Eliminar duplicados por ID.
- Seleccionar sólo camiones [Vans].
- Filtrar por [RatePerMile] (muchas observaciones publican sin tarifa).
- Cambio de Estados de Origen y Destino por zonas (reducción).
- Eliminar lanes intrahub.
- Eliminar outliers generales.

Output:
Dataset para realizar el proyecto.
""")

# Función para cargar y combinar los archivos Parquet
@st.cache_data
def load_data():
    folder_base = os.getcwd()  # Asegúrate de que los archivos .parquet estén en esta carpeta
    parquet_files = glob.glob(os.path.join(folder_base, "*.parquet"))
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df['RatePerMile'] = pd.to_numeric(df['RatePerMile'], errors='coerce')
    cols = ['ID', 'Posted', 'CityOrigin', 'LatOrigin', 'LngOrigin', 'CityDestination',
            'LatDestination', 'LngDestination', 'Size', 'Weight', 'Distance', 'RatePerMile',
            'Equip', 'StateOrigin', 'HubOrigin', 'StateDestination', 'HubDestination']
    df = df[cols]
    # Guarda el dataset combinado (opcional)
    df.to_parquet("loads.parquet")
    return df

# Cargar los datos
df = load_data()
st.write("Dimensiones iniciales del DataFrame:", df.shape)

st.markdown("#### Eliminación de duplicados")
# Se asume que originalmente hay 18625 cargas y se calculan duplicados
duplicados = 18625 - df['ID'].nunique()
st.write("Número de duplicados a eliminar:", duplicados)
df = df.drop_duplicates('ID', keep='first')
st.write("Dimensiones tras eliminar duplicados:", df.shape)

st.markdown("#### Manejo de Origen y Destino por Zona")
st.write("Cantidad de estados únicos en origen y destino:",
         df['StateOrigin'].nunique() + df['StateDestination'].nunique())
st.write("Cantidad de ciudades únicas en origen y destino:",
         df['CityOrigin'].nunique() + df['CityDestination'].nunique())

st.markdown("#### Asignación de Zonas USA")
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
st.write("Zonas de origen disponibles:", df['ZoneOrigin'].unique())

st.markdown("#### Manejo de Camiones")
st.write("Tipos de Equip disponibles inicialmente:", df['Equip'].unique())

def filter_and_explode_equip(data):
    if data.empty:
        return pd.DataFrame()
    desired_values = ['Van', 'Reefer', 'Flatbed']
    column_name = 'Equip'
    desired_values_set = set(desired_values)
    # Convertir de string a lista si es necesario
    data[column_name] = data[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data = data[data[column_name].apply(lambda x: isinstance(x, list))]
    data[column_name] = data[column_name].apply(lambda x: [item for item in x if item in desired_values_set])
    data = data[data[column_name].map(len) > 0]
    data = data.explode(column_name).reset_index(drop=True)
    return data

df = filter_and_explode_equip(df)
st.write("Dimensiones tras filtrar camiones:", df.shape)
st.write("Tipos de Equip filtrados:", df['Equip'].unique())

st.markdown("#### Extracción del Día de la Semana")
df['Posted'] = pd.to_datetime(df['Posted'])
df['weekday'] = df['Posted'].dt.weekday
df['weekday_name'] = df['Posted'].dt.day_name()
pivot = df.pivot_table(index='weekday_name', values='ID', aggfunc='count')
st.write("Conteo de registros por día de la semana:", pivot)

st.markdown("#### Eliminación de registros del día 'Sunday'")
df = df.drop(df.loc[df['weekday_name'] == 'Sunday'].index)
st.write("Dimensiones tras eliminar registros de domingo:", df.shape)

st.markdown("### Fin de la sección de Limpieza de Datos")
st.write("El DataFrame final está listo para análisis posterior.")

# (Opcional) Descargar el dataset limpio en formato CSV
st.markdown("### Descargar Datos Limpios")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar datos en CSV",
    data=csv,
    file_name='datos_limpios.csv',
    mime='text/csv',
)
