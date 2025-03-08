import streamlit as st
import pandas as pd
import numpy as np
import os, glob, ast, io
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import streamlit.components.v1 as components

st.set_page_config(page_title="Análisis de Cargas", layout="wide")

# Título y descripción
st.title("Análisis de Cargas - Proyecto Final")
st.markdown("**Equipo:** Karen Gomez, David Lizcano, Jason Barrios, Camilo Barriosnuevo")

# --- Carga y Limpieza de Datos ---
@st.cache_data
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
df = df[df['weekday_name'] != 'Sunday']  # Remover domingos

# --- Sidebar Resumen ---
st.sidebar.header("Resumen de Datos")
st.sidebar.write("Dimensiones:", df.shape)
st.sidebar.write("Equipos:", df['Equip'].unique())

#########################################
# EDA 1 sin clusters
#########################################

# --- Exploración de Datos ---
st.header("Exploración de Datos")
with st.expander("Información General del DataFrame"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write("Primeras 3 filas:", df.head(3))

st.subheader("Valores Nulos")
st.write(df.isnull().sum())

fig_heat, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='inferno', ax=ax)
ax.set_title("Heatmap de Nulos")
st.pyplot(fig_heat)

# --- Mapa Interactivo ---
st.subheader("Mapa de Cargas (Origen)")
m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
for _, row in df.iterrows():
    color = "blue" if pd.notnull(row["RatePerMile"]) and row["RatePerMile"] > 0 else "orange"
    folium.CircleMarker(location=[row["LatOrigin"], row["LngOrigin"]],
                        radius=2, color=color, fill=True).add_to(m)
components.html(m._repr_html_(), height=500)

# --- Resumen por Tipo de Camión ---
st.subheader("Resumen RatePerMile por Tipo de Camión")
summary = df.groupby('Equip')['RatePerMile'].agg(
    total='size', 
    nulos=lambda x: x.isnull().sum(), 
    no_nulos='count'
)
summary['% nulos'] = (summary['nulos'] / summary['total'] * 100).map("{:.2f}%".format)
st.write(summary)

#########################################
#  Eliminación de RatePerMile NaN y Tratamiento de Outliers
#########################################

import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf

st.header("5. Eliminación de RatePerMile NaN y Tratamiento de Outliers")
st.markdown("""
En esta sección se eliminan los registros sin información de tarifa (RatePerMile ≤ 0), 
ya que la idea es utilizar la información de pago conocida para predecir el comportamiento de pago de aquellas cargas sin tarifa.  
Además, se procede a eliminar outliers que, por distancias cortas o errores de tipeo, generan valores atípicos y ruido en la predicción.
""")

# Se asume que 'df' es el DataFrame base obtenido de "loads.parquet" en bloques anteriores
# Para este bloque, filtramos registros con RatePerMile > 0
df_rates = df.loc[df['RatePerMile'] > 0].copy()

# Pivot table para ver la cantidad de cargas por tipo de camión
piv = df_rates.pivot_table(index='Equip', values='ID', aggfunc='count')
st.subheader("Cargas con RatePerMile definido por tipo de camión")
st.dataframe(piv)
st.write("Total de cargas con RatePerMile:", piv['ID'].sum())

st.markdown("""
Existen solo 4455 cargas con información de RatePerMile en los camiones principales.
""")

# Eliminación de registros donde HubOrigin y HubDestination son iguales (casos de distancias cortas)
st.subheader("Eliminación de outliers por resolución 'Hub'")
num_hub_iguales = len(df_rates.loc[df_rates['HubOrigin'] == df_rates['HubDestination']].index)
st.write("Número de cargas con HubOrigin igual a HubDestination:", num_hub_iguales)

df_rates = df_rates.drop(df_rates[df_rates['HubOrigin'] == df_rates['HubDestination']].index)
st.write("Dimensiones después de eliminar Hub iguales:", df_rates.shape)

st.markdown("""
Ahora, se procede a eliminar outliers que pueden ser resultado de errores de tipeo o input del broker.  
Se utiliza el rango intercuartílico (IQR) de la variable **RatePerMile** como referencia.
""")

def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))
    return data[mask]

df_rates = remove_outliers_iqr(df_rates, 'RatePerMile')
st.write("Dimensiones después de eliminar outliers en RatePerMile:", df_rates.shape)

st.markdown("""
Se observa una distribución normal de la variable **RatePerMile** para las cargas analizadas, 
donde cerca del 50% se encuentran entre 1.94 y 3.1 USD por milla.
""")

# Histograma de RatePerMile
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(df_rates['RatePerMile'], range=(0, 7), bins=100, color='steelblue', edgecolor='white')
ax_hist.set_xlabel('RatePerMile')
ax_hist.set_ylabel('Frecuencia')
ax_hist.set_title('Histograma de RatePerMile')
st.pyplot(fig_hist)

# Boxplot general de RatePerMile (usando Plotly)
fig_box = px.box(df_rates, y="RatePerMile", title="Boxplot de RatePerMile")
st.plotly_chart(fig_box)

st.markdown("## Análisis General por Tipo de Camión")
st.markdown("""
Luego de realizar el análisis gráfico (diagrama de cajas y bigotes), se observa que existen traslapes entre los bigotes y ligeras diferencias en las medias de **RatePerMile** entre los tipos de camión.  
- *Flatbed* presenta una media ligeramente más alta (≈ 2.85 USD)  
- *Van* (≈ 2.48 USD)  
- *Reefer* (≈ 2.29 USD)  
Se plantea la siguiente hipótesis:

**$H_0$=** No hay diferencia significativa en la media de RatePerMile entre los diferentes tipos de camión.  
**$H_1$=** Al menos un tipo de camión tiene una media significativamente diferente.

Se aplicó un análisis de varianzas (ANOVA) para evaluar la hipótesis.
""")

# Boxplot por tipo de camión usando Plotly
fig_box_equip = px.box(df_rates, x="Equip", y="RatePerMile", color="Equip", title="Boxplot de RatePerMile por Tipo de Camión")
st.plotly_chart(fig_box_equip)

# Boxplot por zona de destino y tipo de camión usando Seaborn
fig_sns, ax_sns = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_rates, x='ZoneDestination', y='RatePerMile', hue='Equip', ax=ax_sns)
ax_sns.set_title("Boxplot de RatePerMile por ZoneDestination y Equip")
st.pyplot(fig_sns)

# ANOVA: Evaluación del efecto de Equip sobre RatePerMile
model = smf.ols('RatePerMile ~ Equip', data=df_rates).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
st.subheader("ANOVA: RatePerMile ~ Equip")
st.write(anova_table)

st.markdown("### Análisis de Cargas Publicadas por Día")
st.markdown("""
Se agrupan las cargas por día (usando la fecha de publicación) para conocer la cantidad de cargas publicadas.  
Este análisis puede complementarse con la variable **weekday** para identificar patrones semanales.
""")
# Agrupar por día y contar IDs únicos
cargas_por_dia = df_rates.groupby(df_rates['Posted'].dt.date)['ID'].nunique().reset_index()
cargas_por_dia.rename(columns={'ID': 'Cantidad'}, inplace=True)

fig_bar = px.bar(cargas_por_dia, x='Posted', y='Cantidad',
                 title='Cantidad de Cargas Publicadas por Día',
                 labels={'Posted': 'Día', 'Cantidad': 'Número de Cargas'},
                 color='Cantidad',
                 color_continuous_scale='Greys')
st.plotly_chart(fig_bar)

#########################################
# EDA DataFrame Final y Reporte Exploratorio
#########################################

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import pandas as pd

st.header("6. EDA DataFrame Final")

st.markdown("""
A continuación se presentan diversas visualizaciones y análisis exploratorios que ayudan a comprender el comportamiento de la variable **RatePerMile** en el dataset final:
- **Mapa interactivo:** Todas las cargas se muestran pintadas por tipo de camión y día.
- **Series de tiempo:** Boxplot de *RatePerMile* por día de la semana para evidenciar su efecto.
- **Combinaciones de zonas:** Se muestran las combinaciones de zona de origen y destino más frecuentes.
- **Reporte exploratorio:** Se genera un reporte automatizado (usando *Pandas Profiling*) que reemplaza a Sweetviz.
- **Matriz de correlación:** Se visualiza la correlación lineal entre variables numéricas.
""")

# Para este bloque se asume que la variable df_rates (o similar) contiene el dataset final
# En este ejemplo, usamos df_rates obtenido en bloques anteriores tras eliminar outliers y registros con RatePerMile ≤ 0.
# Si en tu código la variable es distinta, actualízala acorde.
df_final = df_rates.copy()

# --- Mapa interactivo de cargas por tipo de camión y día ---
st.subheader("Mapa de Cargas (por Tipo de Camión y Día)")
fig_map = px.scatter_mapbox(
    df_final,
    lat="LatOrigin",
    lon="LngOrigin",
    color="Equip",           # Colorea según el tipo de camión
    hover_name="ID",
    hover_data=["Posted"],
    zoom=4,
    height=600,
    mapbox_style="open-street-map"
)
st.plotly_chart(fig_map)

# --- Boxplot de RatePerMile por Día de la Semana ---
st.subheader("Boxplot de RatePerMile por Día de la Semana")
# Extraemos el nombre del día de la columna Posted
df_final['weekday'] = pd.to_datetime(df_final['Posted']).dt.day_name()
fig_box_time = px.box(
    df_final,
    x="weekday",
    y="RatePerMile",
    color="Equip",
    title="RatePerMile por Día de la Semana",
    category_orders={"weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]}
)
st.plotly_chart(fig_box_time)

# --- Combinaciones de Zonas Más Frecuentes ---
st.subheader("Combinaciones de Zonas Más Frecuentes")
# Se crea la columna que combina la zona de origen y la de destino
df_final['ZoneCombination'] = df_final['ZoneOrigin'] + '-' + df_final['ZoneDestination']
combination_counts = df_final['ZoneCombination'].value_counts().reset_index()
combination_counts.columns = ['ZoneCombination', 'Count']

fig_bar_zones = px.bar(
    combination_counts.head(10),
    x='ZoneCombination',
    y='Count',
    title='Top 10 Combinaciones de Zonas',
    color='ZoneCombination',
    color_discrete_sequence=px.colors.qualitative.Prism
)
st.plotly_chart(fig_bar_zones)

st.subheader("Boxplot de RatePerMile por Trayecto")
fig_box_zones = px.box(
    df_final,
    x="ZoneCombination",
    y="RatePerMile",
    category_orders={"ZoneCombination": df_final.groupby("ZoneCombination")["RatePerMile"].mean().sort_values(ascending=False).index},
    title="Boxplot de RatePerMile por Trayecto",
    color='ZoneCombination',
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig_box_zones.update_xaxes(tickangle=-45)
st.plotly_chart(fig_box_zones)

# --- Reporte Exploratorio Automatizado (reemplazo de Sweetviz) ---
st.subheader("Reporte Exploratorio Automatizado")
st.markdown("""
Se utiliza **Pandas Profiling** (a través de la librería *ydata_profiling*) para generar un reporte exploratorio detallado de las variables del dataset y su asociación.  
Este reporte ayuda a identificar distribuciones, correlaciones y posibles problemas en los datos.
""")
try:
    from ydata_profiling import ProfileReport
    profile = ProfileReport(df_final, title="Reporte Exploratorio de Cargas", explorative=True)
    report_html = profile.to_html()
    st.components.v1.html(report_html, height=800, scrolling=True)
except ModuleNotFoundError:
    st.error("El módulo 'ydata_profiling' no está instalado. Por favor, instálalo o agrégalo a tu requirements.txt para ver el reporte exploratorio.")

# --- Matriz de Correlación ---
st.subheader("Matriz de Correlación de Variables Numéricas")
df_num = df_final.select_dtypes(include=['number'])
correlation_matrix = df_num.corr()
fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
ax_corr.set_title("Mapa de Calor de la Matriz de Correlación")
st.pyplot(fig_corr)

# --- Mapa de Trayectos por Combinación de Zonas ---
st.subheader("Mapa de Trayectos por Combinación de Zonas")
# Se define una paleta de colores para las combinaciones de zonas
unique_combinations = df_final["ZoneCombination"].unique()
colores = [
    "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "gray", "black", "cyan", "magenta",
    "lime", "teal", "olive", "navy", "maroon", "aquamarine",
    "coral", "fuchsia", "silver", "gold", "indigo", "lavender"
]
paleta_colores = {zone: colores[i % len(colores)] for i, zone in enumerate(unique_combinations)}

mapa_zones = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
for idx, row in df_final.iterrows():
    folium.CircleMarker(
        location=[row["LatOrigin"], row["LngOrigin"]],
        radius=5,
        color=paleta_colores.get(row["ZoneCombination"], "gray"),
        fill=True,
        fill_color=paleta_colores.get(row["ZoneCombination"], "gray"),
        popup=row["ZoneCombination"],
        tooltip=row["ZoneCombination"]
    ).add_to(mapa_zones)
st.markdown("Mapa interactivo de trayectos por combinaciones de zonas:")
st.components.v1.html(mapa_zones._repr_html_(), height=500)

#########################################
# 4. Análisis de Clustering y Zonas Geográficas (usando loads.parquet)
#########################################

import streamlit.components.v1 as components
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.header("4. Análisis de Clustering y Zonas Geográficas")
st.markdown("""
En esta sección se agrupan las ciudades de origen y destino de las cargas mediante el algoritmo DBSCAN, con el objetivo de identificar zonas geográficas que puedan presentar comportamientos tarifarios similares.  
Se evalúan diferentes combinaciones de parámetros y se generan mapas interactivos para visualizar los clusters resultantes, para posteriormente asignar a cada carga una zona geográfica.
""")

# --- Carga de la base de datos "loads.parquet" ---
@st.cache_data
def load_loads():
    return pd.read_parquet('loads.parquet')

df_loads = load_loads()
st.write("Dimensiones de la base de datos loads.parquet:", df_loads.shape)

# --- Generación del DataFrame de ciudades ---
# Se unen las coordenadas de origen y destino para generar un listado único de ciudades
cities = pd.concat(
    [
        df_loads[['ID', 'LatOrigin', 'LngOrigin', 'StateOrigin']].rename(
            columns={'StateOrigin': 'State', 'LatOrigin': 'Lat', 'LngOrigin': 'Lng'}
        ),
        df_loads[['ID', 'LatDestination', 'LngDestination', 'StateDestination']].rename(
            columns={'StateDestination': 'State', 'LatDestination': 'Lat', 'LngDestination': 'Lng'}
        )
    ],
    ignore_index=True
)
cities = cities.drop_duplicates(subset=['Lat', 'Lng'])
st.write("Número de ciudades únicas:", cities.shape[0])
st.write(cities.head())

# --- Función para generar mapas de clusters ---
def generar_mapa_clusters(df, lat_col='Lat', lon_col='Lng', cluster_col='cluster', zoom_start=5):
    mapa = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()], zoom_start=zoom_start)
    
    # Se obtienen los clusters únicos (excluyendo outliers, que tienen etiqueta -1)
    clusters_unicos = df[cluster_col].unique()
    clusters_unicos = clusters_unicos[clusters_unicos != -1]
    
    # Se crea un colormap con suficientes colores
    colors = plt.cm.get_cmap('gist_ncar', len(clusters_unicos))
    cluster_colors = {
        cluster: "#{:02x}{:02x}{:02x}".format(
            int(colors(i / len(clusters_unicos))[0] * 255),
            int(colors(i / len(clusters_unicos))[1] * 255),
            int(colors(i / len(clusters_unicos))[2] * 255)
        )
        for i, cluster in enumerate(clusters_unicos)
    }
    
    # Se dibujan los círculos de cada cluster basados en el centroide y la máxima distancia (radio)
    for cluster in clusters_unicos:
        df_cluster_temp = df[df[cluster_col] == cluster]
        centroide_lat = df_cluster_temp[lat_col].mean()
        centroide_lon = df_cluster_temp[lon_col].mean()
        
        max_dist_mi = max(df_cluster_temp.apply(
            lambda row: great_circle((centroide_lat, centroide_lon), (row[lat_col], row[lon_col])).miles,
            axis=1
        ))
        
        folium.Circle(
            location=[centroide_lat, centroide_lon],
            radius=max_dist_mi * 1609.34,  # Conversión de millas a metros
            color=cluster_colors[cluster],
            fill=True,
            fill_color=cluster_colors[cluster],
            fill_opacity=0.4,
            popup=f"Cluster {cluster} (Radio: {max_dist_mi:.2f} mi)"
        ).add_to(mapa)
    
    # Se dibujan los puntos individuales
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=2,
            color="black",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6
        ).add_to(mapa)
    
    return mapa

# --- Búsqueda de parámetros para DBSCAN ---
st.markdown("#### Evaluación de Modelos de Clustering")
eps_values = np.arange(10/69, 100/69, 0.15)  # Aproximadamente de 10 a 100 millas (convertido a grados)
min_samples_values = np.arange(5, 31, 5)        # Valores de 5 a 30

resultados = []
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cities['cluster'] = dbscan.fit_predict(cities[['Lat', 'Lng']])
        
        num_clusters = len(set(cities['cluster'])) - (1 if -1 in cities['cluster'].values else 0)
        noise_ratio = (cities['cluster'] == -1).sum() / len(cities)
        
        valid_clusters = cities[cities['cluster'] != -1]
        if num_clusters > 1:
            silhouette = silhouette_score(valid_clusters[['Lat', 'Lng']], valid_clusters['cluster'])
            db_index = davies_bouldin_score(valid_clusters[['Lat', 'Lng']], valid_clusters['cluster'])
        else:
            silhouette, db_index = np.nan, np.nan
        
        resultados.append({
            'eps (mi)': eps * 69,  # Convertir grados a millas
            'min_samples': min_samples,
            'num_clusters': num_clusters,
            'silhouette': silhouette,
            'davies_bouldin': db_index,
            'noise_ratio': noise_ratio
        })

df_resultados = pd.DataFrame(resultados).sort_values(by='silhouette', ascending=False)
df_resultados_best = df_resultados.loc[df_resultados['noise_ratio'] < 0.5].head(10)

st.write("Top 10 modelos (con noise_ratio < 0.5):")
st.dataframe(df_resultados_best)

# --- Selección del Mejor Modelo ---
best_model = df_resultados.loc[df_resultados['silhouette'].idxmax()]
st.write("📌 Mejor Modelo (por Silhouette Score):")
st.write(best_model)

dbscan = DBSCAN(eps=best_model['eps (mi)'] / 69, min_samples=int(best_model['min_samples']))
cities['cluster'] = dbscan.fit_predict(cities[['Lat', 'Lng']])
num_clusters = len(set(cities['cluster'])) - (1 if -1 in cities['cluster'].values else 0)
st.write("Número de clusters (excluyendo outliers):", num_clusters)

st.markdown("**Mapa del Mejor Modelo (por Silhouette Score)**")
mapa_best = generar_mapa_clusters(cities)
components.html(mapa_best._repr_html_(), height=500)

# --- Modelo con menor Noise Ratio ---
best_model_noise = df_resultados.loc[df_resultados['noise_ratio'].idxmin()]
st.write("📌 Mejor Modelo (por menor Noise Ratio):")
st.write(best_model_noise)

dbscan = DBSCAN(eps=best_model_noise['eps (mi)'] / 69, min_samples=int(best_model_noise['min_samples']))
cities['cluster'] = dbscan.fit_predict(cities[['Lat', 'Lng']])
num_clusters_noise = len(set(cities['cluster'])) - (1 if -1 in cities['cluster'].values else 0)
st.write("Número de clusters (excluyendo outliers):", num_clusters_noise)

st.markdown("**Mapa del Modelo con menor Noise Ratio**")
mapa_noise = generar_mapa_clusters(cities)
components.html(mapa_noise._repr_html_(), height=500)

# --- Análisis de Outliers ---
st.markdown("#### Análisis de Outliers")
pivot_clusters = cities.pivot_table(index='cluster', values='ID', aggfunc='count')
st.write("Conteo de observaciones por cluster:")
st.write(pivot_clusters)

st.markdown("Mapa de Observaciones sin Cluster (Outliers, cluster -1)")
mapa_outliers = folium.Map(location=[cities['Lat'].mean(), cities['Lng'].mean()], zoom_start=5)
for _, row in cities[cities['cluster'] == -1].iterrows():
    folium.CircleMarker(
        location=[row['Lat'], row['Lng']],
        radius=2,
        color="black",
        fill=True,
        fill_color="blue",
        fill_opacity=0.6,
        tooltip=str(row['ID'])
    ).add_to(mapa_outliers)
components.html(mapa_outliers._repr_html_(), height=500)

st.write("Estados presentes en outliers:")
st.write(cities[cities['cluster'] == -1]['State'].unique())

st.markdown("#### Estados por Cluster")
for i in range(0, 11):
    st.write(f"Estados del cluster {i}:", cities.loc[cities['cluster'] == i]['State'].unique())

# --- Asignación de Zonas Geográficas ---
st.markdown("#### Asignación de Zonas Geográficas")
st.markdown("""
Se asignan zonas geográficas a partir de los estados de cada ciudad.  
Cada cluster se asocia a una zona mediante la siguiente clasificación:
- **Z0:** TX, NM, OK, AR, LA  
- **Z1:** OH, IN, IL, MI, KY, WV  
- **Z2:** NY, NJ, PA, DE, MD, VA, ME, CT, MA, NH, RI, VT  
- **Z3:** FL, AL, GA, SC, NC, MS, TN  
- **Z4:** AZ, CA, NV, WA, UT, OR  
- **Z5:** KS, IA, MN, ND, SD, NE, WI, MO  
- **Z6:** CO, MT, ID, WY
""")

def get_zone(state_code):
    zones = {
        "Z0": {'TX', 'NM', 'OK', 'AR', 'LA'},
        "Z1": {'OH', 'IN', 'IL', 'MI', 'KY', 'WV'},
        "Z2": {'NY', 'NJ', 'PA', 'DE', 'MD', 'VA', 'ME', "CT", "MA", "NH", "RI", "VT"},
        "Z3": {'FL', 'AL', 'GA', 'SC', 'NC', 'MS', 'TN'},
        "Z4": {'AZ', 'CA', 'NV', 'WA', 'UT', 'OR'},
        "Z5": {'KS', 'IA', 'MN', 'ND', 'SD', 'NE', 'WI', 'MO'},
        "Z6": {'CO', 'MT', 'ID', 'WY'}
    }
    for zone, states in zones.items():
        if state_code in states:
            return zone
    return "Unknown"

cities['zone'] = cities['State'].apply(lambda x: get_zone(x))
st.write("Ejemplo de asignación de zona:")
st.write(cities[['Lat', 'Lng', 'State', 'zone']].head())

# --- Fusionar Zonas Geográficas en el Dataset Original ---
cols_zone = ['Lat', 'Lng', 'zone']
df_merged = df_loads.merge(cities[cols_zone],
                           left_on=['LatOrigin', 'LngOrigin'],
                           right_on=['Lat', 'Lng'],
                           how='left').rename(columns={'zone': 'ZoneOrigin'})
df_merged = df_merged.merge(cities[cols_zone],
                           left_on=['LatDestination', 'LngDestination'],
                           right_on=['Lat', 'Lng'],
                           how='left').rename(columns={'zone': 'ZoneDestination'})
df_merged.drop(columns=['Lat_x', 'Lng_x', 'Lat_y', 'Lng_y'], inplace=True)
st.write("Dimensiones del DataFrame fusionado:", df_merged.shape)

# --- Filtrado y Preparación del Dataset Final ---
# Se filtran las cargas con RatePerMile positivo y se eliminan registros con Hub idéntico en origen y destino
test_final = df_merged[df_merged['RatePerMile'] > 0]
st.write("Dimensiones de cargas con RatePerMile > 0:", test_final.shape)

test_final = test_final.drop(test_final[test_final['HubOrigin'] == test_final['HubDestination']].index)
st.write("Dimensiones después de eliminar Hub iguales:", test_final.shape)

# Función para remover outliers en RatePerMile mediante el método IQR
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
    return df[mask]

test_final = remove_outliers_iqr(test_final, 'RatePerMile')
st.write("Dimensiones después de eliminar outliers en RatePerMile:", test_final.shape)

# Se crean combinaciones de zonas, estados y hubs para análisis futuro
test_final['ZoneCombination'] = test_final['ZoneOrigin'] + test_final['ZoneDestination']
test_final['StateCombination'] = test_final['StateOrigin'] + test_final['StateDestination']
test_final['HubCombination'] = test_final['HubOrigin'] + test_final['HubDestination']
st.write("Número de combinaciones de Hub:", test_final['HubCombination'].nunique())

st.markdown("""
**Conclusiones del Análisis de Clustering y Zonas Geográficas:**

- El análisis de clustering mediante DBSCAN permite identificar agrupaciones de ciudades que pueden tener comportamientos tarifarios similares.  
- Se evaluaron distintos modelos balanceando la separación de clusters (silhouette score) y la cantidad de outliers (noise ratio).  
- La asignación final de zonas geográficas se realiza combinando la información de clusters y estados, reduciendo la complejidad en la variable de trayecto.  
- Se identificaron observaciones atípicas (outliers) que podrían ser reasignadas en análisis posteriores.
""")
