import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from IPython.display import display
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sweetviz as sv
import folium
import requests
from shapely.geometry import Point, LineString, Polygon

# Cargas
df = pd.DataFrame()

folder_base = os.getcwd()
parquet_files = glob.glob(folder_base+"/*.parquet")

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
df.to_parquet("loads.parquet")

# Eliminación duplicados
df = df.drop_duplicates('ID',keep='first')

# Manejo de origen y destino por zona
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

df['ZoneOrigin'] = df['StateOrigin'].apply(lambda x: get_zone(x))
df['ZoneDestination'] = df['StateDestination'].apply(lambda x: get_zone(x))

# Manejo de camiones
import ast
def filter_and_explode_equip(data):
    if data.empty:
        return pd.DataFrame()
    desired_values = ['Van', 'Reefer', 'Flatbed']
    column_name = 'Equip'
    desired_values_set = set(desired_values)
    data[column_name] = data[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data = data[data[column_name].apply(lambda x: isinstance(x, list))]
    data[column_name] = data[column_name].apply(lambda x: [item for item in x if item in desired_values_set])
    data = data[data[column_name].map(len) > 0]
    data = data.explode(column_name).reset_index(drop=True)
    return data

df = filter_and_explode_equip(df)

# Extracción día de la semana
df['Posted'] = pd.to_datetime(df['Posted'])
df['weekday'] = df['Posted'].dt.weekday
df['weekday_name'] = df['Posted'].dt.day_name()
df = df.drop(df.loc[df['weekday_name']=='Sunday'].index)

# Análisis Exploratorio de Datos

## Análisis de valores nulos
df.info()
df.head(3)

null_counts = df.isnull().sum()

plt.figure(figsize=(8, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='inferno')
plt.title('Heatmap de Valores Nulos')
plt.xlabel('Columnas')
plt.ylabel('Filas')
plt.show()

filtered_df = df[df['RatePerMile'].isnull()]
sns.histplot(data=filtered_df, x='Equip', hue='Equip')
plt.title('Histograma de valores nulos de RatePerMile por Equip')
plt.xlabel('Equip')
plt.ylabel('Cantidad de valores nulos')
plt.show()

summary = df.groupby('Equip')['RatePerMile'].agg(
    total='size',
    nulos=lambda x: x.isnull().sum(),
    no_nulos='count'
)
summary['% nulos por Equip'] = (summary['nulos'] / summary['total']) * 100
summary['% nulos por Equip'] = summary['% nulos por Equip'].map("{:.2f}%".format)
print(summary)

df.describe()

# Mapa Situación Actual
tate_total_counts = df.groupby('StateOrigin')['RatePerMile'].size()
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
display(summary_df)

# Visualización en mapa
m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["LatOrigin"], row["LngOrigin"]],
        color="Blue" if row["RatePerMile"] > 0 else "orange",
        fill=True,
    ).add_to(m)

m
