import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os, glob, ast, io
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit.components.v1 as components

st.set_page_config(page_title="Análisis de Cargas", layout="wide")

# Cargar datos y modelo
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
df = df[df['weekday_name'] != 'Sunday']

# Páginas de la aplicación
def pagina_generar_carga():
    st.title("Generar Carga")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.subheader("Generar Carga")
        if st.button("Generar Carga"):
            carga = df.sample(1).iloc[0]
            st.session_state["carga"] = carga
        
        if "carga" in st.session_state:
            st.image(f"images/{st.session_state['carga']['Equip']}.png", caption=st.session_state['carga']['Equip'])

    with col2:
        st.subheader("Mapa")
        if "carga" in st.session_state:
            mapa = folium.Map(location=[st.session_state['carga']['LatOrigin'], st.session_state['carga']['LngOrigin']], zoom_start=6)
            folium.Marker([st.session_state['carga']['LatOrigin'], st.session_state['carga']['LngOrigin']], tooltip="Origen").add_to(mapa)
            folium.Marker([st.session_state['carga']['LatDestination'], st.session_state['carga']['LngDestination']], tooltip="Destino").add_to(mapa)
            folium_static(mapa)

    with col3:
        st.subheader("Detalles")
        if "carga" in st.session_state:
            st.write(f"Weight: {st.session_state['carga']['Weight']} lbs")
            st.write(f"Size: {st.session_state['carga']['Size']} cu ft")
            distancia = np.random.randint(100, 500)  # Simulación de distancia
            st.write(f"Distancia: {distancia} km")
            st.session_state["distancia"] = distancia

def pagina_dueno():
    st.title("Vista Dueño del Vehículo")
    if "carga" in st.session_state:
        pagina_generar_carga()
        st.subheader("Valor de la Carga")
        if "distancia" in st.session_state:
            features = [[st.session_state['carga']['Weight'], st.session_state['carga']['Size'], st.session_state['distancia']]]
            pred = np.random.randint(500, 2000)  # Simulación de predicción
            min_value = pred * 0.9
            max_value = pred * 1.1
            st.write(f"Valor mínimo: ${min_value:.2f}")
            st.write(f"Valor máximo: ${max_value:.2f}")
    else:
        st.write("Por favor, genere una carga primero en la página de Generar Carga.")

pagina = st.sidebar.selectbox("Selecciona una página", ["Generar Carga", "Vista Dueño del Vehículo"])

if pagina == "Generar Carga":
    pagina_generar_carga()
elif pagina == "Vista Dueño del Vehículo":
    pagina_dueno()
