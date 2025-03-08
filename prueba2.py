import streamlit as st
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sweetviz as sv
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from sklearn.metrics import silhouette_score, davies_bouldin_score
import folium
import scipy.stats as stats
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import itertools
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from scipy.stats import jarque_bera, shapiro, normaltest
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
import subprocess
import pickle  # Asegúrate de importar pickle aquí
from streamlit_folium import folium_static
import streamlit.components.v1 as components

st.set_page_config(page_title="Gestión de Cargas", layout="wide")

# --- Cargar datos y modelo ---
@st.cache_data
def load_data():
    file_path = "dataset.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        st.error("⚠️ No se encontró el archivo dataset.parquet")
        return pd.DataFrame()

# --- Pestaña 1: Introducción ---
def pagina_introduccion():
    st.title("Introducción")
    st.write("Estimación de los precios de fletes de transporte de mercancias en USA.")

    # --- Visualización del mapa con datos de loads.parquet ---
    st.subheader("Visualización de Cargas en el Mapa")
    
    @st.cache_data
    def load_loads_data():
        file_path = "loads.parquet"
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        else:
            st.error("⚠️ No se encontró el archivo loads.parquet")
            return pd.DataFrame()
    
    df_loads = load_loads_data()
    
    if not df_loads.empty:
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
        for _, row in df_loads.iterrows():
            folium.CircleMarker(
                location=[row["LatOrigin"], row["LngOrigin"]],
                color="Blue" if row["RatePerMile"] > 0 else "Orange",
                fill=True,
            ).add_to(m)
        folium_static(m)
    else:
        st.warning("No hay datos disponibles para mostrar en el mapa.")



import streamlit as st
import os
import streamlit.components.v1 as components


# --- Pestaña 2: Datos Utilizados ---
def pagina_datos():
    st.title("Datos Utilizados")
    st.write("Esta aplicación utiliza datos de **loads.parquet**, que contiene información detallada sobre los envíos de carga.")
    
    @st.cache_data
    def load_loads_data():
        file_path = "loads.parquet"
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        else:
            st.error("⚠️ No se encontró el archivo loads.parquet")
            return pd.DataFrame()
    
    df = load_loads_data()
    
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Valores Nulos en el Dataset")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.isnull(), cbar=False, cmap='inferno', ax=ax)
            ax.set_title('Heatmap de Valores Nulos')
            ax.set_xlabel('Columnas')
            ax.set_ylabel('Filas')
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.subheader("Porcentaje de Envíos con y sin Rate por Estado")
            state_total_counts = df.groupby('StateOrigin')['RatePerMile'].size()
            state_non_null_counts = df.groupby('StateOrigin')['RatePerMile'].count()
            state_null_counts = df[df['RatePerMile'].isnull()].groupby('StateOrigin').size()
        
            summary_df = pd.DataFrame({
                'Envíos sin Rate': state_null_counts,
                'Envíos con Rate': state_non_null_counts,
                'Total_Envíos': state_total_counts
            }).fillna(0).astype(int)
        
            summary_df['% Envíos sin Rate'] = (summary_df['Envíos sin Rate'] / summary_df['Total_Envíos']) * 100
            summary_df['% Envíos con Rate'] = (summary_df['Envíos con Rate'] / summary_df['Total_Envíos']) * 100
            summary_df = summary_df.sort_values(by=['Total_Envíos'], ascending=False)
        
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['red', 'green']
            summary_df[['% Envíos sin Rate', '% Envíos con Rate']].plot(
                kind="bar", stacked=True, color=colors, ax=ax
            )
        
            for i, state in enumerate(summary_df.index):
                y_sin = summary_df.loc[state, '% Envíos sin Rate']
                y_con = summary_df.loc[state, '% Envíos con Rate']
                total_sin = summary_df.loc[state, 'Envíos sin Rate']
                total_con = summary_df.loc[state, 'Envíos con Rate']
        
                if total_sin > 0:
                    ax.text(i, y_sin / 2, f"{y_sin:.1f}%", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
                if total_con > 0:
                    ax.text(i, y_sin + y_con / 2, f"{y_con:.1f}%", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
            ax.set_ylabel("Porcentaje de Envíos")
            ax.set_xlabel("Estado de Origen")
            ax.set_title("Porcentaje de Envíos con y sin Rate por Estado")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.legend(["Sin Rate", "Con Rate"], loc="upper right")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            
            st.pyplot(fig)
            plt.close(fig)

    # Crear una columna única
    col = st.columns([1])  # El valor 1 indica el ancho de la columna
    
    # Columna única
    with col[0]:  # Accedemos al primer (y único) elemento de la lista
        # Mostrar el ícono de Sweetviz
        st.image("images/sweetviz.png", width=50)
    
        # Ruta del archivo HTML en el directorio principal
        html_file_path = "SWEETVIZ_REPORT.html"
        
        # Verificar si el archivo existe antes de cargarlo
        if os.path.exists(html_file_path):
            # Cargar el contenido del archivo HTML
            with open(html_file_path, "r") as file:
                html_content = file.read()
    
            # Insertar el archivo HTML en la app de Streamlit
            components.html(html_content, height=600)  # Ajusta la altura según sea necesario
        else:
            st.warning("El archivo SWEETVIZ_REPORT.html no se encontró.")


# --- Pestaña 3: Modelo de Predicción ---
def pagina_modelo():
    st.title("Modelo de Predicción")
    st.write("El modelo utilizado en esta aplicación es un Random Forest Regressor entrenado para estimar los costos de transporte.")
    st.write("Se ha calculado un **Mean Absolute Percentage Error (MAPE)** de **11.64%**, lo que indica un buen desempeño en la estimación de costos.")
    st.write("El modelo ha sido entrenado con datos reales de transporte y utiliza variables como el tipo de vehículo, la distancia y el peso para hacer las predicciones.")

df = load_data()

@st.cache_data
def load_encoded_data():
    file_path = "Xtest_encoded.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        st.error("⚠️ No se encontró el archivo Xtest_encoded.parquet")
        return pd.DataFrame()

df_encoded = load_encoded_data()

@st.cache_data
def load_model():
    model_path = "random_forest_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)
    else:
        st.error("⚠️ No se encontró el archivo random_forest_model.pkl")
        return None
import streamlit as st
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sweetviz as sv
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from sklearn.metrics import silhouette_score, davies_bouldin_score
import folium
import scipy.stats as stats
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import itertools
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from scipy.stats import jarque_bera, shapiro, normaltest
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
import subprocess
import pickle  # Asegúrate de importar pickle aquí
from streamlit_folium import folium_static
import streamlit.components.v1 as components
model = load_model()

cities = pd.concat(
    [df[['ID', 'LatOrigin', 'LngOrigin','StateOrigin']].rename(columns={'StateOrigin':'State','LatOrigin': 'Lat', 'LngOrigin': 'Lng'}),
        df[['ID', 'LatDestination', 'LngDestination','StateDestination']].rename(columns={'StateDestination':'State','LatDestination': 'Lat', 'LngDestination': 'Lng'})
    ],
    ignore_index=True
)

#analizar ciudades únicas dado que hay ciudades duplicadas por camión, o porque el destino o el origen se repite varias veces
cities = cities.drop_duplicates(subset=['Lat', 'Lng'])

def generar_mapa_clusters(df, lat_col='Lat', lon_col='Lng', cluster_col='cluster', zoom_start=5):
    mapa = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()], zoom_start=zoom_start)

    # Obtener los clusters únicos excluyendo outliers (-1 si aplica)
    clusters_unicos = df[cluster_col].unique()
    clusters_unicos = clusters_unicos[clusters_unicos != -1]

    # Obtener colormap con suficientes colores
    colors = plt.cm.get_cmap('gist_ncar', len(clusters_unicos))

    # Crear un diccionario de colores para cada cluster
    cluster_colors = {
        cluster: "#{:02x}{:02x}{:02x}".format(
            int(colors(i / len(clusters_unicos))[0] * 255),  # Rojo
            int(colors(i / len(clusters_unicos))[1] * 255),  # Verde
            int(colors(i / len(clusters_unicos))[2] * 255)   # Azul
        )
        for i, cluster in enumerate(clusters_unicos)
    }

    # Dibujar círculos de los clusters
    for cluster in clusters_unicos:
        df_cluster = df[df[cluster_col] == cluster]
        centroide_lat = df_cluster[lat_col].mean()
        centroide_lon = df_cluster[lon_col].mean()

        # Calcular radio (máxima distancia al centroide en millas)
        max_dist_mi = max(df_cluster.apply(
            lambda row: great_circle((centroide_lat, centroide_lon), (row[lat_col], row[lon_col])).miles,
            axis=1
        ))

        # Agregar el círculo del cluster
        folium.Circle(
            location=[centroide_lat, centroide_lon],
            radius=max_dist_mi * 1609.34,  # Convertir mi a metros
            color=cluster_colors[cluster],
            fill=True,
            fill_color=cluster_colors[cluster],
            fill_opacity=0.4,
            popup=f"Cluster {cluster} (Radio: {max_dist_mi:.2f} mi)"
        ).add_to(mapa)

    # Dibujar los puntos individuales
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=2,  # Tamaño del punto
            color="black",  # Borde del punto
            fill=True,
            fill_color="blue",  # Color del punto
            fill_opacity=0.6
        ).add_to(mapa)

    return mapa

eps_values = np.arange(10/69, 100/69, 0.15)  # De 10 a 100 millas, paso de 0.15 grados (1 grado son 69 millas)
min_samples_values = np.arange(5, 31, 5)  # De 5 a 30, paso de 5

resultados = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples) #cuantas cargas quieres unir, y cuánto de distancia entre cargas te hace sentido
        cities['cluster'] = dbscan.fit_predict(cities[['Lat', 'Lng']])

        # Contar clusters (excluyendo -1 que es ruido)
        num_clusters = len(set(cities['cluster'])) - (1 if -1 in cities['cluster'].values else 0)
        noise_ratio = (cities['cluster'] == -1).sum() / len(cities)  # Proporción de ruido

        # Calcular métricas solo si hay más de 1 cluster válido
        valid_clusters = cities[cities['cluster'] != -1]
        if num_clusters > 1:
            silhouette = silhouette_score(valid_clusters[['Lat', 'Lng']], valid_clusters['cluster'])
            db_index = davies_bouldin_score(valid_clusters[['Lat', 'Lng']], valid_clusters['cluster'])
        else:
            silhouette, db_index = np.nan, np.nan  # Evitar errores si hay 1 solo cluster

        resultados.append({
            'eps (mi)': eps * 69,  # Convertir de grados a millas
            'min_samples': min_samples,
            'num_clusters': num_clusters,
            'silhouette': silhouette,
            'davies_bouldin': db_index,
            'noise_ratio': noise_ratio
        })

# Convertir a DataFrame y ordenar por Silhouette Score
df_resultados = pd.DataFrame(resultados).sort_values(by='silhouette', ascending=False)
display(df_resultados.loc[df_resultados['noise_ratio']<0.5].head(10))  # Mostrar los mejores 10 resultados

mejor_modelo = df_resultados[df_resultados['silhouette'] == df_resultados['silhouette'].max()].iloc[0]

# Mostrar los parámetros del mejor modelo
print(f"📌 Mejor Modelo:")
print(mejor_modelo)

dbscan = DBSCAN(eps=mejor_modelo['eps (mi)'] / 69, min_samples=int(mejor_modelo['min_samples']))
cities['cluster'] = dbscan.fit_predict(cities[['Lat', 'Lng']])
print("Numero de clusters: ", cities['cluster'].nunique()-1)

mejor_modelo = df_resultados[df_resultados['noise_ratio'] == df_resultados['noise_ratio'].min()].iloc[0]

# Mostrar los parámetros del mejor modelo
print(f"📌 Mejor Modelo:")
print(mejor_modelo)

dbscan = DBSCAN(eps=mejor_modelo['eps (mi)'] / 69, min_samples=int(mejor_modelo['min_samples']))
cities['cluster'] = dbscan.fit_predict(cities[['Lat', 'Lng']])
print("Numero de clusters: ", cities['cluster'].nunique()-1)

generar_mapa_clusters(cities)

with col[0]:
    st.write("### El mapa resultante del último bloque de código:")
    mapa = generar_mapa_clusters(cities)  # Llama a la función para generar el mapa
    components.folium_static(mapa)  # Muestra el mapa en la app de Streamlit

# --- Generar una carga aleatoria ---
def generar_carga():
    if df.empty:
        st.error("⚠️ No hay datos disponibles.")
        return None, None
    carga = df.sample(1)
    return carga.iloc[0], carga.index[0]  # Devolvemos la fila como serie y su índice

# --- Página 1: Generar Carga ---
def pagina_generar_carga():
    st.title("Generar Carga")
    if st.button("Generar Carga"):
        carga, idx = generar_carga()
        if carga is not None:
            st.session_state["carga"] = carga
            st.session_state["carga_idx"] = idx
    
    if "carga" in st.session_state:
        carga = st.session_state["carga"]
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.subheader("Detalles de la Carga")
            st.write(f"**Origen:** {carga['CityOrigin']}")
            st.write(f"**Destino:** {carga['CityDestination']}")
            st.write(f"**Peso:** {carga['Weight']} lbs")
            st.write(f"**Tamaño:** {carga['Size']} cu ft")
            
            equip = carga['Equip'].lower()
            image_path = f"images/{equip}.png"
            if os.path.exists(image_path):
                st.image(image_path, caption=equip)
            else:
                st.warning(f"Imagen no encontrada: {image_path}")
        
        with col2:
            st.subheader("Ruta en Mapa")
            mapa = folium.Map(location=[carga['LatOrigin'], carga['LngOrigin']], zoom_start=6)
            folium.Marker([carga['LatOrigin'], carga['LngOrigin']], tooltip="Origen").add_to(mapa)
            folium.Marker([carga['LatDestination'], carga['LngDestination']], tooltip="Destino").add_to(mapa)
            folium_static(mapa)
        
        with col3:
            st.subheader("Distancia Estimada")
            distancia = np.random.randint(100, 500)
            st.write(f"**Distancia:** {distancia} km")
            st.session_state["distancia"] = distancia

# --- Página 2: Vista Dueño del Vehículo ---
def pagina_dueno():
    st.title("Vista Dueño del Vehículo")
    
    # Verificar si la carga está en el estado
    if "carga" not in st.session_state or "carga_idx" not in st.session_state:
        st.warning("Genera una carga primero en la otra página.")
        return
    
    carga = st.session_state["carga"]
    idx = st.session_state["carga_idx"]
    MAPE = 0.1164  # Valor del Mape del Modelo en Validación
    if model is not None and "distancia" in st.session_state:
        if idx in df_encoded.index:
            features = df_encoded.loc[idx].values.reshape(1, -1)  # Seleccionamos la misma fila en Xtest_encoded
            pred = model.predict(features)[0]
            min_value = pred - (pred * MAPE)
            max_value = pred + (pred * MAPE)
            
            # Organizar la vista en tres columnas
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.subheader("Detalles de la Carga")
                st.write(f"**Origen:** {carga['CityOrigin']}")
                st.write(f"**Destino:** {carga['CityDestination']}")
                st.write(f"**Peso:** {carga['Weight']} lbs")
                st.write(f"**Tamaño:** {carga['Size']} cu ft")
                
                equip = carga['Equip'].lower()
                image_path = f"images/{equip}.png"
                if os.path.exists(image_path):
                    st.image(image_path, caption=equip)
                else:
                    st.warning(f"Imagen no encontrada: {image_path}")
            
            with col2:
                st.subheader("Ruta en Mapa")
                mapa = folium.Map(location=[carga['LatOrigin'], carga['LngOrigin']], zoom_start=6)
                folium.Marker([carga['LatOrigin'], carga['LngOrigin']], tooltip="Origen").add_to(mapa)
                folium.Marker([carga['LatDestination'], carga['LngDestination']], tooltip="Destino").add_to(mapa)
                folium_static(mapa)
            
            with col3:
                st.subheader("Distancia Estimada")
                distancia_km = st.session_state["distancia"]
                distancia_mi = distancia_km * 0.621371
                st.write(f"**Distancia:** {distancia_km} km ({distancia_mi:.2f} mi)")
                
                st.subheader("Estimación de Pago:")
                st.write(f"💰 **mínimo:** ${min_value:.2f} PerMile -> (${ min_value * distancia_mi:.2f} USD)")
                st.write(f"💰 **máximo:** ${max_value:.2f} PerMile -> (${ max_value * distancia_mi:.2f} USD)")
        else:
            st.warning("No se encontró la fila correspondiente en Xtest_encoded.parquet.")
    else:
        st.warning("No se pudo calcular el pago. Asegúrate de que el modelo está cargado y los datos están correctamente procesados.")

# --- Menú de Navegación ---
pagina = st.sidebar.selectbox("Selecciona una página", ["Introducción", "Datos Utilizados", "Modelo de Predicción", "Generar Carga", "Vista Dueño del Vehículo"])
if pagina == "Introducción":
    pagina_introduccion()
elif pagina == "Datos Utilizados":
    pagina_datos()
elif pagina == "Modelo de Predicción":
    pagina_modelo()
elif pagina == "Generar Carga":
    pagina_generar_carga()
elif pagina == "Vista Dueño del Vehículo":
    pagina_dueno()
