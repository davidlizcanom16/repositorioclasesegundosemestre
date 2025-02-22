import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------- Configuración de la Aplicación ----------------------
st.set_page_config(page_title="Clasificación de Especies 🌿", layout="wide")

# ---------------------- Cargar Dataset ----------------------
@st.cache_data
def cargar_datos():
    return px.data.iris()  # Carga el dataset Iris de Plotly

df = cargar_datos()

# ---------------------- Función de Clasificación ----------------------
def clasificar_especie(row):
    if row["petal_length"] < 2:
        return "Setosa 🌸"
    elif row["petal_length"] < 5:
        return "Versicolor 🌿"
    else:
        return "Virginica 🌺"

df["Especie Clasificada"] = df.apply(clasificar_especie, axis=1)

# ---------------------- Sidebar con Filtros ----------------------
st.sidebar.title("🔍 Filtros")
eje_x = st.sidebar.selectbox("Selecciona el eje X", df.columns[:4])
eje_y = st.sidebar.selectbox("Selecciona el eje Y", df.columns[:4])
color = st.sidebar.selectbox("Color por", ["species", "Especie Clasificada"])

# ---------------------- Gráfico Interactivo ----------------------
st.title("📊 Clasificación de Especies por Dimensiones")
st.write("Explora cómo se distribuyen las especies de Iris según las dimensiones de sus pétalos y sépalos.")

fig = px.scatter(df, x=eje_x, y=eje_y, color=color, hover_data=df.columns, 
                 title="Distribución de Especies", template="plotly_dark", 
                 color_discrete_map={"Setosa 🌸": "blue", "Versicolor 🌿": "green", "Virginica 🌺": "red"})

st.plotly_chart(fig, use_container_width=True)

# ---------------------- Mostrar Datos ----------------------
st.subheader("📋 Datos del Dataset")
st.dataframe(df)
