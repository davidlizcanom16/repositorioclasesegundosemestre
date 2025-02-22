import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos de ejemplo (puedes cambiar esto por tu propio dataset)
df = sns.load_dataset("iris")

# Título de la aplicación
st.title("📊 Análisis de Datos con Streamlit")

# Mostrar la tabla de datos
st.subheader("📄 Datos")
st.write(df)

# Matriz de correlación
st.subheader("📊 Mapa de Calor de Correlaciones")

# Filtrar solo columnas numéricas
corr_matrix = df.select_dtypes(include=['number']).corr()

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)

# Mostrar gráfico en Streamlit
st.pyplot(fig)
