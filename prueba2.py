import pandas as pd
import plotly.express as px
import streamlit as st
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import numpy as np
from scipy.stats import pearsonr

# Función para calcular el correlation ratio (correlación entre variable categórica y numérica)
def correlation_ratio(categories, values):
    categories = categories.astype(str)
    category_means = values.groupby(categories).mean()
    overall_mean = values.mean()
    numerator = ((category_means - overall_mean) ** 2 * values.groupby(categories).count()).sum()
    denominator = ((values - overall_mean) ** 2).sum()
    return np.sqrt(numerator / denominator) if denominator != 0 else 0

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
df = df[[col for col in cols if col in df.columns]]

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

# Análisis de correlaciones
st.header("Análisis de correlaciones")

# Correlaciones numéricas
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
correlation_matrix = df[numeric_columns].corr()
st.write("Matriz de correlaciones:")
st.write(correlation_matrix)

# Gráfico de correlaciones
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(plt)

# Correlaciones categóricas
categorical_columns = ['HubDestination', 'HubOrigin', 'StateDestination', 'StateOrigin', 'Equip', 'weekday_name']
categorical_columns = [col for col in categorical_columns if col in df.columns]

correlation_ratios = {col: correlation_ratio(df[col], df['RatePerMile']) for col in categorical_columns}
correlation_ratios_df = pd.DataFrame(list(correlation_ratios.items()), columns=['Categorical Variable', 'Correlation Ratio'])
st.write("Correlaciones Categóricas (Correlation Ratio):")
st.write(correlation_ratios_df)

# Multicolinealidad - VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df, features):
    X = df[features].dropna()
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(features))]
    return vif_data

if len(numeric_columns) > 1:
    vif_df = calculate_vif(df, numeric_columns)
    st.write("Factor de Inflación de Varianza (VIF) para detectar multicolinealidad:")
    st.write(vif_df)
else:
    st.write("No hay suficientes variables numéricas para calcular el VIF.")

# Visualización de correlaciones con gráficos interactivos
fig = px.scatter_matrix(df, dimensions=numeric_columns, title="Matriz de Dispersión de Variables Numéricas")
st.plotly_chart(fig)
