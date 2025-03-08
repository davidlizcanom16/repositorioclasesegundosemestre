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
# 4. Análisis de Clustering y Zonas Geográficas
#########################################

st.header("4. Análisis de Clustering y Zonas Geográficas")
st.markdown("""
En esta sección se busca agrupar las ciudades de origen y destino de las cargas mediante DBSCAN, con el fin de identificar zonas geográficas que puedan presentar comportamientos tarifarios similares.  
Se evalúan diferentes combinaciones de parámetros y se generan mapas interactivos para visualizar los clusters resultantes.
""")

# Cargar datos de clusters (se asume que el archivo 'loadsclusters.parquet' está en la misma carpeta)
@st.cache_data
def load_cluster_data():
    return pd.read_parquet('loadsclusters.parquet')

df_cluster = load_cluster_data()
st.write("Dimensiones de los datos de clusters:", df_cluster.shape)

# Generar DataFrame de ciudades (origen y destino)
cities = pd.concat(
    [
        df_cluster[['ID', 'LatOrigin', 'LngOrigin', 'StateOrigin']].rename(
            columns={'StateOrigin': 'State', 'LatOrigin': 'Lat', 'LngOrigin': 'Lng'}
        ),
        df_cluster[['ID', 'LatDestination', 'LngDestination', 'StateDestination']].rename(
            columns={'StateDestination': 'State', 'LatDestination': 'Lat', 'LngDestination': 'Lng'}
        )
    ],
    ignore_index=True
)
cities = cities.drop_duplicates(subset=['Lat', 'Lng'])
st.write("Número de ciudades únicas:", cities.shape[0])
st.write(cities.head())

# Función para generar mapas de clusters
def generar_mapa_clusters(df, lat_col='Lat', lon_col='Lng', cluster_col='cluster', zoom_start=5):
    mapa = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()], zoom_start=zoom_start)
    
    # Obtener clusters únicos (excluyendo outliers: -1)
    clusters_unicos = df[cluster_col].unique()
    clusters_unicos = clusters_unicos[clusters_unicos != -1]
    
    # Crear colormap con suficientes colores
    colors = plt.cm.get_cmap('gist_ncar', len(clusters_unicos))
    cluster_colors = {
        cluster: "#{:02x}{:02x}{:02x}".format(
            int(colors(i / len(clusters_unicos))[0] * 255),
            int(colors(i / len(clusters_unicos))[1] * 255),
            int(colors(i / len(clusters_unicos))[2] * 255)
        )
        for i, cluster in enumerate(clusters_unicos)
    }
    
    # Dibujar círculos para cada cluster
    for cluster in clusters_unicos:
        df_cluster_temp = df[df[cluster_col] == cluster]
        centroide_lat = df_cluster_temp[lat_col].mean()
        centroide_lon = df_cluster_temp[lon_col].mean()
        
        # Calcular radio (máxima distancia al centroide en millas)
        max_dist_mi = max(df_cluster_temp.apply(
            lambda row: great_circle((centroide_lat, centroide_lon), (row[lat_col], row[lon_col])).miles,
            axis=1
        ))
        
        folium.Circle(
            location=[centroide_lat, centroide_lon],
            radius=max_dist_mi * 1609.34,  # Convertir millas a metros
            color=cluster_colors[cluster],
            fill=True,
            fill_color=cluster_colors[cluster],
            fill_opacity=0.4,
            popup=f"Cluster {cluster} (Radio: {max_dist_mi:.2f} mi)"
        ).add_to(mapa)
    
    # Dibujar puntos individuales
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

# Búsqueda de parámetros para DBSCAN
eps_values = np.arange(10/69, 100/69, 0.15)  # Aproximadamente de 10 a 100 millas (convertido a grados)
min_samples_values = np.arange(5, 31, 5)  # Valores de 5 a 30

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

st.subheader("Resultados de Clustering")
st.write("Top 10 modelos (con noise_ratio < 0.5):")
st.dataframe(df_resultados_best)

# Seleccionar el mejor modelo según el silhouette score
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

# Modelo con 27 clusters (si existe alguno)
if 27 in df_resultados['num_clusters'].values:
    best_model_27 = df_resultados[df_resultados['num_clusters'] == 27].iloc[0]
    st.write("📌 Mejor Modelo con 27 clusters:")
    st.write(best_model_27)
    
    dbscan = DBSCAN(eps=best_model_27['eps (mi)'] / 69, min_samples=int(best_model_27['min_samples']))
    cities['cluster'] = dbscan.fit_predict(cities[['Lat', 'Lng']])
    num_clusters_27 = len(set(cities['cluster'])) - (1 if -1 in cities['cluster'].values else 0)
    st.write("Número de clusters (excluyendo outliers):", num_clusters_27)
    
    st.markdown("**Mapa del Modelo con 27 clusters**")
    mapa_27 = generar_mapa_clusters(cities)
    components.html(mapa_27._repr_html_(), height=500)

# Modelo con menor noise_ratio
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

st.markdown("#### Análisis de Outliers y Estados en Outliers")
# Pivot table de conteo de observaciones por cluster
pivot_clusters = cities.pivot_table(index='cluster', values='ID', aggfunc='count')
st.write("Conteo de observaciones por cluster:")
st.write(pivot_clusters)

# Mapa de outliers (cluster -1)
st.markdown("Mapa de Observaciones sin Cluster (Outliers)")
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

st.markdown("#### Asignación de Zonas Geográficas")
st.markdown("""
A partir de los clusters obtenidos se asignan zonas geográficas.  
Cada cluster se asocia a una zona en función de los estados que comprende.
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
st.write("Ejemplo de asignación de zona para algunas ciudades:")
st.write(cities[['Lat', 'Lng', 'State', 'zone']].head())

# Fusionar la información de zonas en el DataFrame original
cols_zone = ['Lat', 'Lng', 'zone']
df_merged = df_cluster.merge(cities[cols_zone],
                             left_on=['LatOrigin', 'LngOrigin'],
                             right_on=['Lat', 'Lng'],
                             how='left').rename(columns={'zone': 'ZoneOrigin'})
df_merged = df_merged.merge(cities[cols_zone],
                             left_on=['LatDestination', 'LngDestination'],
                             right_on=['Lat', 'Lng'],
                             how='left').rename(columns={'zone': 'ZoneDestination'})
df_merged.drop(columns=['Lat_x', 'Lng_x', 'Lat_y', 'Lng_y'], inplace=True)
st.write("Dimensiones del DataFrame fusionado:", df_merged.shape)

# Filtrar datos para análisis final: cargas con RatePerMile > 0
test_final = df_merged[df_merged['RatePerMile'] > 0]
st.write("Dimensiones de cargas con RatePerMile > 0:", test_final.shape)

# Eliminar registros donde HubOrigin es igual a HubDestination
test_final = test_final.drop(test_final[test_final['HubOrigin'] == test_final['HubDestination']].index)
st.write("Dimensiones después de eliminar Hub iguales:", test_final.shape)

# Función para remover outliers usando el método IQR
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
    return df[mask]

test_final = remove_outliers_iqr(test_final, 'RatePerMile')
st.write("Dimensiones después de eliminar outliers en RatePerMile:", test_final.shape)

# Crear combinaciones de zonas y estados
test_final['ZoneCombination'] = test_final['ZoneOrigin'] + test_final['ZoneDestination']
test_final['StateCombination'] = test_final['StateOrigin'] + test_final['StateDestination']
test_final['HubCombination'] = test_final['HubOrigin'] + test_final['HubDestination']
st.write("Número de combinaciones de Hub:", test_final['HubCombination'].nunique())

st.markdown("""
**Conclusiones del Análisis de Clustering y Zonas Geográficas:**

- El análisis de clustering mediante DBSCAN permite identificar agrupaciones de ciudades que pueden tener comportamientos tarifarios similares.
- Se evaluaron distintos modelos balanceando la separación de clusters (silhouette score) y la cantidad de outliers (noise ratio).
- La asignación final de zonas geográficas se realiza combinando la información de clusters y estados, reduciendo la complejidad en la variable de trayecto.
- Se identificaron observaciones atípicas (outliers) que podrían ser reasignadas a clusters específicos en análisis posteriores.
""")
